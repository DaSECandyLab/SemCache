"""Apply an updated search-timeout policy to an already-running worker."""

import json
import os
from pathlib import Path
import signal
import time

from experiments.core.io import save

RUN = Path('/home/lyp/project/experiments/results/system_comparison/compare-20')
TASK = 'abacus-MinTime-q50/tpch/Q17'
PID = 3392998
FOLDER = RUN / 'tasks' / TASK / 'attempt-1'


def main():
    while True:
        status = json.loads((RUN / 'status.json').read_text())
        if status.get('task_id') != TASK or status.get('child_pid') != PID:
            return
        if not Path(f'/proc/{PID}').exists():
            return
        for path in (FOLDER / 'native-regions').glob('region-*/mechanism.json'):
            try:
                evidence = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if evidence.get('fallback_used') and evidence.get('search_timeout'):
                status = json.loads((RUN / 'status.json').read_text())
                if status.get('task_id') != TASK or status.get('child_pid') != PID:
                    return
                if (FOLDER / 'summary.json').exists():
                    return
                save(FOLDER / 'summary.json', {
                    'status': 'timeout', 'timed_out': True, 'timeout_scope': 'plan_search',
                    'reason': 'Abacus plan search exceeded 100 seconds; fallback stopped under the active policy.',
                })
                try:
                    os.killpg(PID, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                return
        time.sleep(0.2)


if __name__ == '__main__':
    main()
