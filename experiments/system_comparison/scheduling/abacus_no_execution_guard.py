"""Skip the active campaign's Abacus tasks stuck before any model request."""

from datetime import datetime
import json
import os
from pathlib import Path
import re
import signal
import time

from experiments.core.io import save

RUN = Path('/home/lyp/project/experiments/results/system_comparison/compare-20')
CONTROLLER = 3096960
SAMPLE_LIMIT_SECONDS = 100
EVENT = re.compile(
    r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+) INFO '
    r'palimpzest\.query\.execution\.mab_execution_strategy '
    r'(Executing plan|Done executing sentinel plan)'
)


def last_sampling_event(path):
    try:
        with path.open('rb') as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - 262144))
            lines = stream.read().decode(errors='replace').splitlines()
    except OSError:
        return None
    events = [EVENT.match(line) for line in lines]
    events = [event for event in events if event]
    if not events or events[-1].group(2) != 'Executing plan':
        return None
    return datetime.strptime(events[-1].group(1), '%Y-%m-%d %H:%M:%S,%f').timestamp()


def main():
    while Path(f'/proc/{CONTROLLER}').exists():
        try:
            status = json.loads((RUN / 'status.json').read_text())
            task, pid = status.get('task_id', ''), status.get('child_pid')
            if (status.get('phase') != 'running' or not task.startswith('abacus-MinTime-q50/')
                    or not isinstance(pid, int) or status.get('model_requests') != 0):
                time.sleep(2)
                continue
            folder = RUN / status['folder']
            sampling_started = last_sampling_event(folder / 'native.log')
            gateway = folder / 'gateway.jsonl'
            if (sampling_started is None or time.time() - sampling_started < SAMPLE_LIMIT_SECONDS
                    or (gateway.exists() and gateway.stat().st_size > 0)):
                time.sleep(2)
                continue
            current = json.loads((RUN / 'status.json').read_text())
            if (current.get('task_id') != task or current.get('child_pid') != pid
                    or current.get('model_requests') != 0 or not Path(f'/proc/{pid}').exists()
                    or (gateway.exists() and gateway.stat().st_size > 0)
                    or (folder / 'summary.json').exists()):
                continue
            if os.getpgid(pid) != pid:
                raise RuntimeError('Worker is not the expected process-group leader')
            save(folder / 'summary.json', {
                'status': 'skipped', 'timed_out': False,
                'skip_classification': 'zero_requests_stalled_sampling',
                'reason': 'MAB sampling exceeded 100 seconds without any model request or final plan execution; skipped per user request.',
                'sampling_without_requests_seconds': time.time() - sampling_started,
            })
            os.killpg(pid, signal.SIGKILL)
            print(f'SKIPPED {task}: no model requests after 100 seconds in MAB sampling', flush=True)
        except (FileNotFoundError, ProcessLookupError, json.JSONDecodeError):
            pass
        time.sleep(2)


if __name__ == '__main__':
    main()
