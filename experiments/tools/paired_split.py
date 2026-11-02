"""Paired dPPG of a new run against reference runs, split by map type (fixed course maps vs RANDOM maps).

    venv/bin/python -B experiments/tools/paired_split.py NEW.jsonl OLD.jsonl [OLD2.jsonl ...]
Like paired.py (a game is opponent, layout, colour; several OLD runs are averaged), printed for all maps, the fixed
maps and the random maps.
"""
import math
import sys

from paired import load


def stats(dp):
    n = len(dp)
    if n < 2:
        return 'n=%d' % n
    m = sum(dp) / n
    se = math.sqrt(sum((x - m) ** 2 for x in dp) / (n - 1) / n)
    return 'n=%d dPPG=%+.3f se=%.3f z=%.1f' % (n, m, se, m / se if se else 0.0)


def main():
    new = load(sys.argv[1])
    olds = [load(f) for f in sys.argv[2:]]
    keys = [k for k in new if all(k in o for o in olds)]
    diff = {k: new[k] - sum(o[k] for o in olds) / len(olds) for k in keys}
    print('all %s | fixed %s | random %s' % (
        stats(list(diff.values())),
        stats([v for k, v in diff.items() if not k[1].startswith('RANDOM')]),
        stats([v for k, v in diff.items() if k[1].startswith('RANDOM')])))


if __name__ == '__main__':
    main()
