"""Continue the not-yet-planned MOAR queries after the forced plan-reuse run."""

import fcntl
import json
from pathlib import Path
import subprocess
import time

from experiments.core.io import save

ROOT = Path('/home/lyp/project')
RUN = ROOT / 'experiments/results/system_comparison/compare-20'
PY = ROOT / 'SemCache/.venv-optimizer/bin/python'
STATE = RUN / 'moar-force-rerun.json'
REUSE_CONTROLLER = 155108
REUSE = [
    'estate/Q5', 'estate/Q9', 'estate/Q10', 'estate/Q11', 'estate/Q12',
    'movies/Q9', 'movies/Q10', 'movies/Q11', 'movies/Q12', 'movies_sema/Q10',
]
REMAINING = [
    'appstore/Q7', 'appstore/Q8', 'appstore/Q9', 'tpch/Q16', 'tpch/Q17',
    'tpch/Q25', 'tpch/Q26', 'tpch/Q27', 'tpch/Q28', 'tpch/Q30',
]


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0] != 'Z'
    except (OSError, IndexError):
        return False


def state(phase, **kwargs):
    info = {'phase': phase, 'reused_queries': REUSE, 'remaining_queries': REMAINING,
            'reuse_controller_pid': REUSE_CONTROLLER, 'updated_unix': time.time(), **kwargs}
    save(STATE, info)
    print(json.dumps(info, ensure_ascii=False), flush=True)


def main():
    try:
        state('running_reused_plans')
        while alive(REUSE_CONTROLLER):
            time.sleep(10)
        with (RUN / 'campaign.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            current = json.loads((RUN / 'status.json').read_text())
            if current.get('phase') != 'repeat_pending' or current.get('controller_pid') != REUSE_CONTROLLER:
                raise RuntimeError('Forced plan-reuse run did not finish normally; not starting remaining queries')
            control = json.loads((RUN / 'control.json').read_text())
            if ('moar' in control.get('disabled_systems', [])
                    or 'moar-pool' in control.get('disabled_variants', [])
                    or any('moar-pool/' + q in control.get('skipped_tasks', []) for q in REMAINING)):
                raise RuntimeError('MOAR or a remaining query was explicitly stopped/skipped')
            for query in REUSE:
                path = RUN / 'tasks' / 'moar-pool' / query / 'attempt-1/summary.json'
                if not path.exists() or not json.loads(path.read_text()).get('parent_finalized'):
                    raise RuntimeError(f'Plan-reuse task did not finish: {query}')
        state('starting_remaining')
        with (RUN / 'moar-force-remaining.log').open('ab') as log:
            child = subprocess.Popen(
                [str(PY), '-u', '-m', 'experiments.system_comparison.scheduling.rerun',
                 '--run', str(RUN), '--variant', 'moar-pool', '--queries', *REMAINING,
                 '--skip-cache-reset', '--overwrite'],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        state('running_remaining', controller_pid=child.pid)
        code = child.wait()
        if code:
            raise RuntimeError(f'Remaining MOAR controller failed with exit code {code}')
        state('finished')
    except Exception as error:
        state('failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
