"""Rerun pre-policy MOAR queries by reusing their selected plans."""

import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from experiments.core.io import save

ROOT = Path('/home/lyp/project')
RUN = ROOT / 'experiments/results/system_comparison/compare-20'
PY = ROOT / 'SemCache/.venv-optimizer/bin/python'
STATE = RUN / 'moar-compact-rerun.json'
OLD_CONTROLLER = 6029
QUERIES = [
    'estate/Q5', 'estate/Q9', 'estate/Q10', 'estate/Q11', 'estate/Q12',
    'movies/Q9', 'movies/Q10', 'movies/Q11', 'movies/Q12',
]


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0] != 'Z'
    except (OSError, IndexError):
        return False


def state(phase, **kwargs):
    result = {'phase': phase, 'queries': QUERIES, 'updated_unix': time.time(), **kwargs}
    save(STATE, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def start_monitor_if_needed():
    marker = RUN / 'monitor-launch.json'
    try:
        pid = json.loads(marker.read_text()).get('pid')
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
        state('waiting_for_current_moar', controller_pid=OLD_CONTROLLER)
        while alive(OLD_CONTROLLER):
            time.sleep(10)
        with (RUN / 'campaign.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            control = json.loads((RUN / 'control.json').read_text())
            if ('moar' in control.get('disabled_systems', [])
                    or 'moar-pool' in control.get('disabled_variants', [])):
                raise RuntimeError('MOAR was stopped; deferred rerun will not start')
            if any('moar-pool/' + query in control.get('skipped_tasks', []) for query in QUERIES):
                raise RuntimeError('A queued MOAR query was explicitly skipped; deferred rerun will not start')
            config = json.loads((RUN / 'campaign.json').read_text())
            tasks = [task for task in config['tasks'] if task['variant'] == 'moar-pool']
            if len(tasks) != 20:
                raise RuntimeError('Unexpected MOAR workload size')
            for task in tasks:
                path = RUN / task['folder'] / 'summary.json'
                if not path.exists() or not json.loads(path.read_text()).get('parent_finalized'):
                    raise RuntimeError(f"Original MOAR run not finalized: {task['task_id']}")
            for query in ('estate/Q11', 'estate/Q12'):
                path = RUN / 'tasks' / 'moar-pool' / query / 'attempt-1/summary.json'
                record = json.loads(path.read_text())
                if record.get('status') != 'failed' or 'InvalidOutputError' not in record.get('reason', ''):
                    raise RuntimeError(f'{query} no longer has the expected failed result')
        state('starting')
        with (RUN / 'moar-compact-rerun.log').open('ab') as log:
            child = subprocess.Popen(
                [str(PY), '-u', '-m', 'experiments.system_comparison.scheduling.rerun',
                 '--run', str(RUN), '--variant', 'moar-pool', '--queries', *QUERIES,
                 '--reuse-moar-selected-plans', '--overwrite'],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        state('starting', rerun_pid=child.pid)
        for _ in range(120):
            if child.poll() is not None:
                raise RuntimeError(f'Rerun exited before starting: {child.returncode}')
            current = json.loads((RUN / 'status.json').read_text())
            if current.get('phase') == 'running' and current.get('controller_pid') == child.pid:
                start_monitor_if_needed()
                break
            time.sleep(1)
        else:
            raise RuntimeError('Rerun did not report a running query within 120 seconds')
        state('running', rerun_pid=child.pid)
        code = child.wait()
        if code:
            raise RuntimeError(f'Rerun controller exited with code {code}')
        statuses = {}
        for query in QUERIES:
            path = RUN / 'tasks' / 'moar-pool' / query / 'attempt-1/summary.json'
            statuses[query] = json.loads(path.read_text()).get('status')
        state('finished', statuses=statuses)
    except Exception as error:
        state('failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
