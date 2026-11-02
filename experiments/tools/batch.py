"""Run a batch of diagnostic matches in parallel; both colours per (team, opponent, layout, seed).

    venv/bin/python -B local-runs/lab/batch.py OUT.jsonl --teams A.py B.py --opps L2.py \
        --layouts defaultCapture RANDOM1 ... --seeds 1 2 --jobs 8
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
FIXED = ['defaultCapture', 'alleyCapture', 'bloxCapture', 'crowdedCapture', 'distantCapture', 'fastCapture',
         'jumboCapture', 'mediumCapture', 'officeCapture', 'strategicCapture']
SETS = {'fixed10': FIXED,
        'rand10': [f'RANDOM{i}' for i in range(1, 11)],
        'rand30': [f'RANDOM{i}' for i in range(1, 31)],
        'dev20': FIXED + [f'RANDOM{i}' for i in range(1, 11)],
        'dev40': FIXED + [f'RANDOM{i}' for i in range(1, 31)],
        'hold30': [f'RANDOM{i}' for i in range(5001, 5031)]}
PY = sys.executable


ENV = dict(os.environ)


def run_one(job):
    red, blue, layout, seed = job
    cmd = [PY, '-B', os.path.join(HERE, 'match.py'), red, blue, layout, str(seed)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=ENV).stdout
        line = [l for l in out.splitlines() if l.startswith('{')][-1]
        return json.loads(line)
    except Exception as exc:  # keep the batch going; record the failure
        return dict(red=red, blue=blue, layout=layout, seed=seed, error=repr(exc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--teams', nargs='+', required=True)
    ap.add_argument('--opps', nargs='+', required=True)
    ap.add_argument('--layouts', nargs='+', required=True)
    ap.add_argument('--seeds', nargs='+', type=int, default=[1])
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--params', default=None, help='JSON passed to agents as PACLAB_PARAMS')
    a = ap.parse_args()
    if a.params:
        # Only the first team under test reads the override (opponent copies of our code stay at defaults).
        ENV['PACLAB_PARAMS'] = a.params
        ENV['PACLAB_PARAMS_FILE'] = os.path.abspath(a.teams[0])
    jobs = []
    layouts = [x for name in a.layouts for x in SETS.get(name, [name])]
    for t in a.teams:
        for o in a.opps:
            for lay in layouts:
                for s in a.seeds:
                    jobs.append((t, o, lay, s))
                    jobs.append((o, t, lay, s))
    with open(a.out, 'a') as f, ThreadPoolExecutor(a.jobs) as pool:
        for n, res in enumerate(pool.map(run_one, jobs), 1):
            f.write(json.dumps(res) + '\n')
            f.flush()
            if n % 20 == 0:
                print(f'{n}/{len(jobs)}', flush=True)


if __name__ == '__main__':
    main()
