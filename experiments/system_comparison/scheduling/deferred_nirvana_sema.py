"""Finish the existing Abacus run, then run the requested comparison sequence."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.request

from experiments.core.io import save


ROOT = Path('/home/lyp/project')
RUN = ROOT / 'experiments/results/system_comparison/compare-20'
PYTHON = ROOT / 'SemCache/.venv-optimizer/bin/python'
STATE = RUN / 'nirvana-sema-deferred.json'
SSH_HOST = 'ccyang@49.52.27.20'
OLD_REMOTE_SCRIPT_SHA = '9c772e9192fe3dcbc9a1ae7b6a70af664e0860eebd3e1a6104c00229e481a515'


def state(phase, **details):
    record = {'phase': phase, 'updated_unix': time.time(), **details}
    save(STATE, record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def wait_for_abacus():
    state('waiting_for_abacus', controller_pid=3096960)
    while True:
        with (RUN / 'campaign.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                time.sleep(15)
                continue
            config = json.loads((RUN / 'campaign.json').read_text())
            tasks = [t for t in config['tasks'] if t['variant'] == 'abacus-MinTime-q50']
            unfinished = []
            for task in tasks:
                path = RUN / task['folder'] / 'summary.json'
                if not path.exists() or not json.loads(path.read_text()).get('parent_finalized'):
                    unfinished.append(task['task_id'])
            if len(tasks) != 20 or unfinished:
                raise RuntimeError(f'Abacus controller exited without finishing: {unfinished}')
            return


def ssh_environment():
    source = (ROOT / 'sglang/ssh_tunnels.sh').read_text()
    match = re.search(r"SSHPASS='([^']+)'", source)
    if not match:
        raise RuntimeError('SSH tunnel credential source unavailable')
    environment = os.environ.copy()
    environment['SSHPASS'] = match.group(1)
    return environment


def server_info():
    with urllib.request.urlopen('http://127.0.0.1:30003/get_server_info', timeout=10) as response:
        return json.load(response)


def ensure_graph():
    info = server_info()
    if info.get('served_model_name') != 'gemma4-12b':
        raise RuntimeError('Port 30003 is not the expected 12B model')
    if info.get('cuda_graph_backend_decode') != 'full' or info.get('disable_cuda_graph'):
        state('restarting_12b_for_cuda_graph')
        env = ssh_environment()
        local = ROOT / 'sglang/remote/start_models.sh'
        remote_temp = '/data/homedata/ycc/sglang-runtime/start_models.codex-graph.sh'
        common = ['-o', 'StrictHostKeyChecking=no', '-o', 'ConnectTimeout=10']
        subprocess.run(['sshpass', '-e', 'scp', *common, str(local), f'{SSH_HOST}:{remote_temp}'], env=env, check=True)
        command = f'''set -euo pipefail
script=/data/homedata/ycc/sglang-runtime/start_models.sh
test "$(sha256sum "$script" | cut -d' ' -f1)" = "{OLD_REMOTE_SCRIPT_SHA}"
pid=$(< /data/homedata/ycc/sglang-runtime/logs/gemma4-12b.pid)
[[ "$pid" =~ ^[0-9]+$ ]]
args=$(ps -p "$pid" -o args=)
[[ "$args" == *sglang.launch_server* && "$args" == *"--served-model-name gemma4-12b"* && "$args" == *"--port 30003"* ]]
[[ "$(ps -p "$pid" -o pgid= | tr -d ' ')" == "$pid" ]]
install -m 755 {remote_temp} "$script"
kill -TERM -- "-$pid"
for i in {{1..90}}; do
  if ! ss -Hlt 'sport = :30003' | grep -q .; then break; fi
  sleep 1
done
if ss -Hlt 'sport = :30003' | grep -q .; then echo '12B did not stop' >&2; exit 1; fi
"$script" ada
'''
        subprocess.run(['sshpass', '-e', 'ssh', *common, SSH_HOST, 'bash', '-s'], input=command, text=True, env=env, check=True)
        info = server_info()
    if (info.get('served_model_name') != 'gemma4-12b'
            or info.get('cuda_graph_backend_decode') != 'full'
            or info.get('disable_cuda_graph')
            or info.get('context_length') != 65536
            or info.get('disable_radix_cache')):
        raise RuntimeError('12B CUDA Graph/context/cache verification failed')
    state('cuda_graph_verified', model='gemma4-12b', decode='full', context_length=65536,
          prefix_cache=True)


def configure():
    path = RUN / 'campaign.json'
    config = json.loads(path.read_text())
    config.setdefault('system_concurrency', {}).update(nirvana=16, sema=16)
    config.setdefault('system_gateway_limits', {}).update(nirvana=16, sema=16)
    config.setdefault('system_prompt_cache', {}).update(nirvana=True, sema=True)
    config.setdefault('system_batch_sizes', {})['sema'] = 4
    config['prompt_cache_enabled'] = True
    save(path, config)
    control_path = RUN / 'control.json'
    control = json.loads(control_path.read_text())
    control.update(preserve_cache=False, query_timeout_seconds=1800,
                   reason='MOAR deferred; rerun Nirvana and Sema with 16 concurrency and prompt cache')
    rerun = {'nirvana-pool', 'sema-e2b', 'sema-e4b', 'sema-12b'}
    control['disabled_variants'] = [v for v in control.get('disabled_variants', []) if v not in rerun]
    save(control_path, control)


def main():
    try:
        wait_for_abacus()
        ensure_graph()
        configure()
        variants = ['nirvana-pool', 'sema-e2b', 'sema-e4b', 'sema-12b']
        for variant in variants:
            state('running', variant=variant, variants=variants)
            subprocess.run([str(PYTHON), '-u', '-m', 'experiments.system_comparison.scheduling.rerun',
                            '--run', str(RUN), '--variant', variant, '--overwrite'], cwd=ROOT, check=True)
        state('finished', variants=variants)
    except BaseException as error:
        state('failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
