"""Variant x opponent matrix from several batch files (team = the file's non-opponent side).

    venv/bin/python -B experiments/tools/compare.py out/a.jsonl out/b.jsonl ...
Cell: win% / points-per-game (3 win, 1 draw) / mean margin, n games. Last column pools all opponents.
"""
import collections
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from match import name_of  # noqa: E402


def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows if 'error' not in r]
    # The team under test is the side that appears in every row.
    counts = collections.Counter()
    for r in rows:
        counts[r['red']] += 1
        counts[r['blue']] += 1
    team = counts.most_common(1)[0][0]
    res = collections.defaultdict(list)
    for r in rows:
        sign = 1 if r['red'] == team else -1
        opp = r['blue'] if sign == 1 else r['red']
        res[name_of(opp)].append(r['score'] * sign)
    return name_of(team), res


def cell(scores):
    n = len(scores)
    if not n:
        return '-'
    w = sum(s > 0 for s in scores)
    d = sum(s == 0 for s in scores)
    ppg = (3 * w + d) / n
    return f'{100 * w / n:3.0f}% {ppg:.2f} {sum(scores) / n:+5.1f} n{n}'


def main():
    tables = [(os.path.basename(p).replace('.jsonl', ''),) + load(p) for p in sys.argv[1:]]
    opps = sorted({o for _, _, res in tables for o in res})
    short = {o: o.replace('aljaz_', '').replace('_my_team.py', '')[:16] for o in opps}
    print(f"{'variant':24}" + ''.join(f'{short[o]:>24}' for o in opps) + f"{'ALL':>26}")
    for label, team, res in tables:
        pooled = [s for o in opps for s in res.get(o, [])]
        w = sum(s > 0 for s in pooled)
        n = len(pooled)
        se = math.sqrt(max(1e-9, (w / n) * (1 - w / n) / n)) if n else 0
        print(f'{label[:24]:24}' + ''.join(f'{cell(res.get(o, [])):>24}' for o in opps) +
              f'{cell(pooled):>22} +-{100 * se:.0f}')


if __name__ == '__main__':
    main()
