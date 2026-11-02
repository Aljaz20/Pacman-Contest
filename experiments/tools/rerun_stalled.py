"""Replay-free repair of a results file: rerun every game with a crash, timeout or an agent time above 200 s.

    venv/bin/python -B experiments/tools/rerun_stalled.py IN.jsonl OUT.jsonl [JOBS]
A machine stall (sleep) shows up as agent times of hundreds of seconds and forfeits both ways. The current code
must play like the recorded one (check with replay_check.py); runs without PACLAB_PARAMS.
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))


def stalled(r):
    return r.get('crashed') or r.get('timeout') or 'error' in r or max(r.get('agent_time') or [0]) > 200


def main():
    rows = [json.loads(line) for line in open(sys.argv[1])]
    jobs = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    bad = [i for i, r in enumerate(rows) if stalled(r)]

    def run(i):
        r = rows[i]
        cmd = [sys.executable, '-B', os.path.join(HERE, 'match.py'), r['red'], r['blue'], r['layout'], str(r['seed'])]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=900).stdout
        return i, json.loads([line for line in out.splitlines() if line.startswith('{')][-1])

    with ThreadPoolExecutor(jobs) as pool:
        for i, new in pool.map(run, bad):
            rows[i] = new
    with open(sys.argv[2], 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    print('rerun %d games, still bad %d' % (len(bad), sum(1 for r in rows if stalled(r))))


if __name__ == '__main__':
    main()
