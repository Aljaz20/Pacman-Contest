"""Round robin: every pair of teams on every layout in both colours; resumable and ordered layout by layout.

    venv/bin/python -B experiments/tools/tournament.py OUT.jsonl TEAMS.txt --layouts defaultCapture RANDOM9001 ... --jobs 10
TEAMS.txt holds one team path per line. Games already in OUT (same red, blue, layout, seed, without an error) are
skipped, so a stopped run continues where it ended. Each layout is finished for all pairs before the next one
starts, so a partial run is still balanced. Games run through match.py under tournament limits.
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))


def run_one(job):
    red, blue, layout, seed = job
    cmd = [sys.executable, '-B', os.path.join(HERE, 'match.py'), red, blue, layout, str(seed)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=900).stdout
        return json.loads([line for line in out.splitlines() if line.startswith('{')][-1])
    except Exception as exc:  # keep the tournament going; the failure is recorded and rerun on resume
        return dict(red=red, blue=blue, layout=layout, seed=seed, error=repr(exc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('teams')
    ap.add_argument('--layouts', nargs='+', required=True)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--jobs', type=int, default=10)
    a = ap.parse_args()
    teams = [line.strip() for line in open(a.teams) if line.strip() and not line.startswith('#')]
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            if 'error' not in r:
                done.add((r['red'], r['blue'], r['layout'], r['seed']))
    jobs = [(red, blue, lay, a.seed) for lay in a.layouts for x, y in itertools.combinations(teams, 2)
            for red, blue in ((x, y), (y, x)) if (red, blue, lay, a.seed) not in done]
    total, start = len(jobs), time.time()
    print('teams %d layouts %d games to play %d (done %d)' % (len(teams), len(a.layouts), total, len(done)), flush=True)
    with ThreadPoolExecutor(a.jobs) as pool, open(a.out, 'a') as f:
        futures = [pool.submit(run_one, j) for j in jobs]
        for k, fut in enumerate(as_completed(futures), 1):
            f.write(json.dumps(fut.result()) + '\n')
            f.flush()
            if k % 500 == 0 or k == total:
                rate = k / (time.time() - start) * 3600
                print('%d / %d games, %.0f per hour, %.1f h left' % (k, total, rate, (total - k) / rate), flush=True)


if __name__ == '__main__':
    main()
