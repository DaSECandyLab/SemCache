"""Resume TPC-H queries temporarily deferred for MOAR Q8."""

import fcntl
import json
from pathlib import Path
import subprocess
import time

from experiments.core.io import save

ROOT = Path('/home/lyp/project')
RUN = ROOT / 'experiments/results/system_comparison/compare-20'
PY = ROOT / 'SemCache/.venv-optimizer/bin/python'
STATE = RUN / 'moar-after-q8.json'
Q8_CONTROLLER = 412510
PRIORITY_REASON = 'Prioritize MOAR appstore/Q8 after Q9; temporarily defer TPC-H'
DEFERRED = [
    'moar-pool/tpch/Q17', 'moar-pool/tpch/Q25', 'moar-pool/tpch/Q26',
    'moar-pool/tpch/Q27', 'moar-pool/tpch/Q28', 'moar-pool/tpch/Q30',
]
QUERIES = ['tpch/Q16'] + [task.removeprefix('moar-pool/') for task in DEFERRED]


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0] != 'Z'
    except (OSError, IndexError):
        return False


def state(phase, **kwargs):
    info = {'phase': phase, 'queries': QUERIES, 'q8_controller_pid': Q8_CONTROLLER,
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
        state('waiting_for_q8')
        while alive(Q8_CONTROLLER):
            time.sleep(5)
        with (RUN / 'campaign.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            current = json.loads((RUN / 'status.json').read_text())
            if current.get('phase') != 'repeat_pending' or current.get('controller_pid') != Q8_CONTROLLER:
                raise RuntimeError('Q8 rerun did not finish normally; TPC-H not resumed')
            q8 = json.loads((RUN / 'tasks/moar-pool/appstore/Q8/attempt-1/summary.json').read_text())
            if not q8.get('parent_finalized'):
                raise RuntimeError('Q8 result is not finalized; TPC-H not resumed')
            control_path = RUN / 'control.json'
            control = json.loads(control_path.read_text())
            if ('moar' in control.get('disabled_systems', [])
                    or 'moar-pool' in control.get('disabled_variants', [])):
                raise RuntimeError('MOAR was explicitly stopped; TPC-H not resumed')
            if (control.get('reason') != PRIORITY_REASON
                    or control.get('q8_priority_deferred_tasks') != DEFERRED):
                raise RuntimeError('Campaign controls changed; TPC-H not resumed automatically')
            if any(task not in control.get('skipped_tasks', []) for task in DEFERRED):
                raise RuntimeError('Deferred-task controls changed; refusing to overwrite them')
            control['skipped_tasks'] = [task for task in control['skipped_tasks'] if task not in DEFERRED]
            control.pop('q8_priority_deferred_tasks', None)
            control.pop('skip_reason', None)
            control['reason'] = 'Resume MOAR TPC-H after the prioritized Q8 rerun'
            save(control_path, control)
        state('starting', q8_result=q8.get('status'))
        with (RUN / 'moar-after-q8.log').open('ab') as log:
            child = subprocess.Popen(
                [str(PY), '-u', '-m', 'experiments.system_comparison.scheduling.rerun',
                 '--run', str(RUN), '--variant', 'moar-pool', '--queries', *QUERIES,
                 '--skip-cache-reset', '--overwrite'],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        for _ in range(120):
            if child.poll() is not None:
                raise RuntimeError(f'TPC-H controller exited before starting: {child.returncode}')
            current = json.loads((RUN / 'status.json').read_text())
            if current.get('phase') == 'running' and current.get('controller_pid') == child.pid:
                start_monitor_if_needed()
                break
            time.sleep(1)
        else:
            raise RuntimeError('TPC-H did not report a running query within 120 seconds')
        state('running', controller_pid=child.pid, q8_result=q8.get('status'))
        code = child.wait()
        if code:
            raise RuntimeError(f'TPC-H controller exited with code {code}')
        state('finished', q8_result=q8.get('status'))
    except Exception as error:
        state('failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
