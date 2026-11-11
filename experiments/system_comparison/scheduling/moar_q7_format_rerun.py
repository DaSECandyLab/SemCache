"""Rerun MOAR Appstore Q7 with its saved plan after the active queue."""

import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time

from experiments.core.io import save

ROOT = Path('/home/lyp/project')
RUN = ROOT / 'experiments/results/system_comparison/compare-20'
PY = ROOT / 'SemCache/.venv-optimizer/bin/python'
STATE = RUN / 'moar-q7-format-rerun.json'
CURRENT_CONTROLLER = 218026
FOLDER = RUN / 'tasks/moar-pool/appstore/Q7/attempt-1'
SOURCE_SHA = hashlib.sha256((FOLDER / 'selected-pipeline.yaml').read_bytes()).hexdigest()


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0] != 'Z'
    except (OSError, IndexError):
        return False


def state(phase, **kwargs):
    info = {'phase': phase, 'query': 'appstore/Q7', 'source_plan_sha256': SOURCE_SHA,
            'updated_unix': time.time(), **kwargs}
    save(STATE, info)
    print(json.dumps(info, ensure_ascii=False), flush=True)


def start_monitor_if_needed():
    try:
        pid = json.loads((RUN / 'monitor-launch.json').read_text()).get('pid')
        cmdline = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ')
        if alive(pid) and b'experiments.system_comparison.reporting.monitor' in cmdline:
            return
    except (OSError, ValueError):
        pass
    with (RUN / 'monitor.log').open('ab') as log:
        subprocess.Popen([str(PY), '-u', '-m', 'experiments.system_comparison.reporting.monitor',
                          '--run', str(RUN)], cwd=ROOT, stdin=subprocess.DEVNULL,
                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def main():
    try:
        state('waiting_for_current_moar', controller_pid=CURRENT_CONTROLLER)
        while alive(CURRENT_CONTROLLER):
            time.sleep(10)
        with (RUN / 'campaign.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            current = json.loads((RUN / 'status.json').read_text())
            if current.get('phase') != 'selection_finished' or current.get('controller_pid') != CURRENT_CONTROLLER:
                raise RuntimeError('Current MOAR controller did not finish normally')
            control = json.loads((RUN / 'control.json').read_text())
            if ('moar' in control.get('disabled_systems', [])
                    or 'moar-pool' in control.get('disabled_variants', [])
                    or 'moar-pool/appstore/Q7' in control.get('skipped_tasks', [])):
                raise RuntimeError('MOAR or Q7 was explicitly stopped/skipped')
            record = json.loads((FOLDER / 'summary.json').read_text())
            if (record.get('status') != 'failed' or 'InvalidOutputError' not in record.get('reason', '')
                    or hashlib.sha256((FOLDER / 'selected-pipeline.yaml').read_bytes()).hexdigest() != SOURCE_SHA):
                raise RuntimeError('Q7 result or selected plan changed; refusing to overwrite')
        state('starting')
        with (RUN / 'moar-q7-format-rerun.log').open('ab') as log:
            child = subprocess.Popen(
                [str(PY), '-u', '-m', 'experiments.system_comparison.scheduling.rerun',
                 '--run', str(RUN), '--variant', 'moar-pool', '--queries', 'appstore/Q7',
                 '--reuse-moar-selected-plans', '--skip-cache-reset', '--overwrite'],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        state('starting', rerun_pid=child.pid)
        for _ in range(120):
            if child.poll() is not None:
                raise RuntimeError(f'Q7 rerun exited before starting: {child.returncode}')
            current = json.loads((RUN / 'status.json').read_text())
            if current.get('phase') == 'running' and current.get('controller_pid') == child.pid:
                start_monitor_if_needed()
                break
            time.sleep(1)
        else:
            raise RuntimeError('Q7 rerun did not report a running query within 120 seconds')
        state('running', rerun_pid=child.pid)
        code = child.wait()
        if code:
            raise RuntimeError(f'Q7 rerun controller exited with code {code}')
        result = json.loads((FOLDER / 'summary.json').read_text())
        state('finished', result_status=result.get('status'))
    except Exception as error:
        state('failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
