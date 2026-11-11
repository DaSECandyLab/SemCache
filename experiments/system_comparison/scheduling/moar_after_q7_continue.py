"""Resume the MOAR work temporarily deferred for Q7 priority."""

import fcntl
import json
from pathlib import Path
import subprocess
import time

from experiments.core.io import save

ROOT = Path('/home/lyp/project')
RUN = ROOT / 'experiments/results/system_comparison/compare-20'
PY = ROOT / 'SemCache/.venv-optimizer/bin/python'
STATE = RUN / 'moar-after-q7.json'
PRIORITY_REASON = 'Prioritize MOAR appstore/Q7 after Q8; temporarily defer Q9 and TPC-H'
DEFERRED = [
    'moar-pool/tpch/Q16', 'moar-pool/tpch/Q17', 'moar-pool/tpch/Q25',
    'moar-pool/tpch/Q26', 'moar-pool/tpch/Q27', 'moar-pool/tpch/Q28', 'moar-pool/tpch/Q30',
]
QUERIES = ['appstore/Q9'] + [task.removeprefix('moar-pool/') for task in DEFERRED]


def state(phase, **kwargs):
    info = {'phase': phase, 'queries': QUERIES, 'updated_unix': time.time(), **kwargs}
    save(STATE, info)
    print(json.dumps(info, ensure_ascii=False), flush=True)


def start_monitor_if_needed():
    try:
        pid = json.loads((RUN / 'monitor-launch.json').read_text()).get('pid')
        cmdline = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ')
        if b'experiments.system_comparison.reporting.monitor' in cmdline:
            return
    except (OSError, ValueError):
        pass
    with (RUN / 'monitor.log').open('ab') as log:
        subprocess.Popen([str(PY), '-u', '-m', 'experiments.system_comparison.reporting.monitor',
                          '--run', str(RUN)], cwd=ROOT, stdin=subprocess.DEVNULL,
                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def main():
    try:
        state('waiting_for_q7')
        while True:
            q7 = json.loads((RUN / 'moar-q7-format-rerun.json').read_text())
            if q7.get('phase') == 'failed':
                raise RuntimeError('Q7 rerun controller failed; deferred queries not resumed')
            if q7.get('phase') == 'finished':
                break
            time.sleep(5)
        with (RUN / 'campaign.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            control_path = RUN / 'control.json'
            control = json.loads(control_path.read_text())
            if ('moar' in control.get('disabled_systems', [])
                    or 'moar-pool' in control.get('disabled_variants', [])):
                raise RuntimeError('MOAR was explicitly stopped; deferred queries not resumed')
            if (control.get('reason') != PRIORITY_REASON
                    or control.get('q7_priority_deferred_tasks') != DEFERRED):
                raise RuntimeError('Campaign controls changed; deferred queries not resumed automatically')
            if any(task not in control.get('skipped_tasks', []) for task in DEFERRED):
                raise RuntimeError('Deferred-task controls changed; refusing to overwrite them')
            control['skipped_tasks'] = [task for task in control['skipped_tasks'] if task not in DEFERRED]
            control.pop('q7_priority_deferred_tasks', None)
            control.pop('skip_reason', None)
            control['reason'] = 'Resume MOAR Q9 and TPC-H after the prioritized Q7 rerun'
            save(control_path, control)
        state('starting')
        with (RUN / 'moar-after-q7.log').open('ab') as log:
            child = subprocess.Popen(
                [str(PY), '-u', '-m', 'experiments.system_comparison.scheduling.rerun',
                 '--run', str(RUN), '--variant', 'moar-pool', '--queries', *QUERIES,
                 '--skip-cache-reset', '--overwrite'],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        for _ in range(120):
            if child.poll() is not None:
                raise RuntimeError(f'Deferred MOAR controller exited before starting: {child.returncode}')
            current = json.loads((RUN / 'status.json').read_text())
            if current.get('phase') == 'running' and current.get('controller_pid') == child.pid:
                start_monitor_if_needed()
                break
            time.sleep(1)
        else:
            raise RuntimeError('Deferred MOAR did not report a running query within 120 seconds')
        state('running', controller_pid=child.pid, q7_result=q7.get('result_status'))
        code = child.wait()
        if code:
            raise RuntimeError(f'Deferred MOAR controller exited with code {code}')
        state('finished', q7_result=q7.get('result_status'))
    except Exception as error:
        state('failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
