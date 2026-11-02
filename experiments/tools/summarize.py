"""Summarise batch JSONL from the perspective of each 'team' (non-opponent) file.

    venv/bin/python -B local-runs/lab/summarize.py OUT.jsonl [--focus substring]
"""
import collections
import json
import os
import sys


def name(path):
    return os.path.basename(os.path.dirname(path)) if path.endswith('my_team.py') else os.path.basename(path)


def main():
    rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    agg = collections.defaultdict(lambda: collections.Counter())
    per_layout = collections.defaultdict(lambda: collections.Counter())
    errors = 0
    for r in rows:
        if 'error' in r:
            errors += 1
            continue
        for colour, sign, mine in (('red', 1, (0, 2)), ('blue', -1, (1, 3))):
            me, opp = name(r[colour]), name(r['blue' if colour == 'red' else 'red'])
            key = (me, opp)
            s = r['score'] * sign
            c = agg[key]
            c['games'] += 1
            c['win'] += s > 0
            c['draw'] += s == 0
            c['loss'] += s < 0
            c['margin'] += s
            c['crash'] += r['crashed']
            c['timeout'] += r['timeout']
            c['warn'] += r['warnings']
            theirs = [i for i in range(4) if i not in mine]
            for i in mine:
                st = r['stats'][i]
                c['deaths'] += st['deaths']
                c['lost'] += st['lost']
                c['kills'] += st['kills']
                c['returned'] += st['returned']
                c['pac_turns'] += st['pac_turns']
                c['capsules'] += st['capsules']
            for i in theirs:
                st = r['stats'][i]
                c['opp_deaths'] += st['deaths']
                c['opp_lost'] += st['lost']
                c['opp_returned'] += st['returned']
                c['opp_pac_turns'] += st['pac_turns']
            pl = per_layout[(me, opp, r['layout'])]
            pl['games'] += 1
            pl['win'] += s > 0
            pl['loss'] += s < 0
            pl['margin'] += s
    print(f'rows={len(rows)} errors={errors}')
    hdr = 'team vs opp | G W D L | margin | deaths lost kills ret pac% caps | oppDeaths oppLost oppRet oppPac% | crash tout warn'
    print(hdr)
    for (me, opp), c in sorted(agg.items()):
        focus = [a for a in sys.argv[2:] if not a.startswith('--')]
        if focus and focus[0] not in me:
            continue
        g = c['games']
        print(f"{me[:34]:34} vs {opp[:14]:14} | {g} {c['win']} {c['draw']} {c['loss']} "
              f"({100*c['win']/g:.0f}%) | {c['margin']/g:+.1f} | "
              f"{c['deaths']/g:.1f} {c['lost']/g:.1f} {c['kills']/g:.1f} {c['returned']/g:.1f} "
              f"{100*c['pac_turns']/(600*g):.0f}% {c['capsules']/g:.1f} | "
              f"{c['opp_deaths']/g:.1f} {c['opp_lost']/g:.1f} {c['opp_returned']/g:.1f} "
              f"{100*c['opp_pac_turns']/(600*g):.0f}% | {c['crash']} {c['timeout']} {c['warn']}")
    if '--layouts' in sys.argv:
        for (me, opp, lay), c in sorted(per_layout.items()):
            print(f"  {me[:30]:30} vs {opp[:12]:12} {lay:16} {c['games']} W{c['win']} L{c['loss']} m{c['margin']/c['games']:+.1f}")


if __name__ == '__main__':
    main()
