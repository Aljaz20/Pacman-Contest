"""Paired dPPG of a new run against one or more runs of the old version over identical games.

    venv/bin/python -B experiments/tools/paired.py NEW.jsonl OLD.jsonl [OLD2.jsonl ...]
A game is (opponent, layout, colour); with several OLD runs the new result is compared to their mean, which
matters against time-bounded opponents that do not replay the same game twice.
"""
import collections
import json
import math
import sys

from report import key_of


def load(path):
    """{(opponent, layout, we_are_red): points} for the team that appears in every row."""
    rows = [json.loads(line) for line in open(path)]
    me = (collections.Counter(r['red'] for r in rows) + collections.Counter(r['blue'] for r in rows)).most_common(1)[0][0]
    out = {}
    for r in rows:
        red = r['red'] == me
        opp = key_of(r['blue'] if red else r['red'])
        if opp == key_of(me):
            continue
        sc = r['score'] if red else -r['score']
        out[(opp, r['layout'], red)] = 3 if sc > 0 else 1 if sc == 0 else 0
    return out


def main():
    new = load(sys.argv[1])
    olds = [load(f) for f in sys.argv[2:]]
    keys = [k for k in new if all(k in o for o in olds)]
    dp = [new[k] - sum(o[k] for o in olds) / len(olds) for k in keys]
    n = len(dp)
    m = sum(dp) / n
    se = math.sqrt(sum((x - m) ** 2 for x in dp) / (n - 1) / n)
    print('n=%d dPPG=%+.3f se=%.3f z=%.1f' % (n, m, se, m / se if se else 0.0))


if __name__ == '__main__':
    main()
