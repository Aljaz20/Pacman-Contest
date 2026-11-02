"""Build the tournament page: standings, head-to-head matrix, paired comparisons of our versions, data checks.

    venv/bin/python -B experiments/tools/tournament_report.py GAMES.jsonl TEAMS.txt OUT.html
Points: win 3, draw 1, loss 0. Rows with a runner error are left out and counted in the checks.
"""
import collections
import html
import json
import math
import sys

from report import key_of, name_of

OURS = {'aljaz2_final': 'Ours: final', 'aljaz2_prev': 'Ours: previous final (help_two 1)',
        'aljaz2_v1p': 'Ours: v1p (milestone)', 'aljaz2_v1a': 'Ours: v1a (first)'}

def label(key):
    return OURS.get(key) or name_of(key)


def points(sc):
    return 3 if sc > 0 else 1 if sc == 0 else 0


def mean_se(xs):
    n = len(xs)
    m = sum(xs) / n
    se = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n) if n > 1 else 0.0
    return m, se


def cell_class(ppg):
    return 'good' if ppg >= 2.0 else 'mid' if ppg >= 1.0 else 'bad'


def main():
    games_path, teams_path, out = sys.argv[1:4]
    teams = [key_of(line.strip()) for line in open(teams_path) if line.strip() and not line.startswith('#')]
    rows = [json.loads(line) for line in open(games_path)]
    errors = [r for r in rows if 'error' in r]
    rows = [r for r in rows if 'error' not in r]
    pts = collections.defaultdict(list)                   # team -> points per game
    kind_pts = collections.defaultdict(list)              # (team, 'fixed' / 'random') -> points
    diff = collections.defaultdict(list)                  # team -> score difference per game
    h2h = collections.defaultdict(list)                   # (team, opp) -> points
    res = {}                                              # (team, opp, layout, team_is_red) -> points
    faults = collections.Counter()                        # (team, kind)
    for r in rows:
        red, blue = key_of(r['red']), key_of(r['blue'])
        for me, opp, sign in ((red, blue, 1), (blue, red, -1)):
            p = points(sign * r['score'])
            pts[me].append(p)
            kind_pts[(me, 'random' if r['layout'].startswith('RANDOM') else 'fixed')].append(p)
            diff[me].append(sign * r['score'])
            h2h[(me, opp)].append(p)
            res[(me, opp, r['layout'], sign == 1)] = p
        times = r.get('agent_time') or [0, 0, 0, 0]
        if max(times) > 200:
            faults[('machine', 'stall')] += 1
        if r.get('crashed') or r.get('timeout'):
            # The engine does not say which side failed; the side with the larger agent time is the usual culprit.
            side = red if times[0] + times[2] >= times[1] + times[3] else blue
            faults[(side, 'crash or timeout')] += 1
        if r.get('warnings'):
            faults[('all', 'time warnings')] += r['warnings']
    order = sorted(teams, key=lambda t: -sum(pts[t]) / max(1, len(pts[t])))
    layouts = sorted({r['layout'] for r in rows})
    expected = len(teams) * (len(teams) - 1) * len(layouts)

    def tag(text, cls=None):
        return '<td%s>%s</td>' % (' class="%s"' % cls if cls else '', text)

    h = ['<!doctype html><html lang="en"><head><meta charset="utf-8">',
         '<meta name="viewport" content="width=device-width, initial-scale=1"><title>PacLab Tournament</title>',
         STYLE, '</head><body><main>',
         '<h1>PacLab Tournament</h1>',
         '<p class="muted">Round robin of %d teams: every pair on %d maps in both colours, %d games (%d expected). '
         'Win 3 points, draw 1, loss 0. Course Python 3.8.20 with tournament limits (1 s per move with 3 warnings, '
         '3 s single move, 15 s setup), 10 games in parallel.</p>' % (len(teams), len(layouts), len(rows), expected)]
    h.append('<h2>Standings</h2><div class="scroll"><table class="res"><tr><th>#</th><th>Team</th><th>Games</th>'
             '<th>Win</th><th>Draw</th><th>Loss</th><th>Win %</th><th>Points / game</th><th>Fixed maps</th>'
             '<th>Random maps</th><th>Score diff / game</th></tr>')
    for k, t in enumerate(order, 1):
        p = pts[t]
        m, se = mean_se(p)
        w, d = sum(1 for x in p if x == 3), sum(1 for x in p if x == 1)
        cls = ' class="ours"' if t in OURS else ''
        h.append('<tr%s><td>%d</td><td>%s</td><td>%d</td><td>%d</td><td>%d</td><td>%d</td><td>%.0f%%</td>'
                 '<td><b>%.2f</b><span>&plusmn; %.2f</span></td><td>%s</td><td>%s</td><td>%+.1f</td></tr>'
                 % (cls, k, html.escape(label(t)), len(p), w, d, len(p) - w - d, 100.0 * w / len(p), m, se,
                    *['%.2f' % (sum(kind_pts[(t, kd)]) / len(kind_pts[(t, kd)])) if kind_pts[(t, kd)] else '-'
                      for kd in ('fixed', 'random')], sum(diff[t]) / len(diff[t])))
    h.append('</table></div><p class="muted">&plusmn; is one standard error of the points per game. Fixed maps are the '
             'classic course layouts, random maps are new RANDOM layouts never used in tuning.</p>')

    h.append('<h2>Head to head</h2><p class="muted">Points per game of the row team against the column team '
             '(%d games per pair). Green &ge; 2.0, yellow &ge; 1.0, red below.</p>' % (2 * len(layouts)))
    h.append('<div class="scroll"><table class="res mat"><tr><th></th>%s</tr>'
             % ''.join('<th title="%s">%d</th>' % (html.escape(label(t)), k) for k, t in enumerate(order, 1)))
    for k, t in enumerate(order, 1):
        cells = []
        for o in order:
            p = h2h.get((t, o))
            cells.append('<td class="na"></td>' if t == o or not p else tag('%.2f' % (sum(p) / len(p)),
                                                                              cell_class(sum(p) / len(p))))
        h.append('<tr><th>%d. %s</th>%s</tr>' % (k, html.escape(label(t)), ''.join(cells)))
    h.append('</table></div>')

    h.append('<h2>Our versions, paired</h2><p class="muted">Difference in points per game over identical games '
             '(same opponent, map and colour), against every opponent both teams played.</p>')
    h.append('<table class="res"><tr><th>Comparison</th><th>Games</th><th>Difference</th></tr>')
    pairs = [('aljaz2_final', 'aljaz2_prev'), ('aljaz2_final', 'aljaz2_v1p'), ('aljaz2_final', 'aljaz2_v1a'),
             ('aljaz2_final', 'aljaz_astar_v27_full_state')]
    for a, b in pairs:
        keys = [(o, lay, red) for (t, o, lay, red) in res if t == a and o not in (a, b) and (b, o, lay, red) in res]
        if not keys:
            continue
        m, se = mean_se([res[(a,) + k] - res[(b,) + k] for k in keys])
        h.append('<tr><td>%s minus %s</td><td>%d</td><td class="%s">%+.2f <span>&plusmn; %.2f</span></td></tr>'
                 % (html.escape(label(a)), html.escape(label(b)), len(keys), 'up' if m > 0 else 'down', m, se))
    h.append('</table>')

    h.append('<h2>Our final by opponent</h2><table class="res"><tr><th>Opponent</th><th>Win %</th>'
             '<th>Points / game</th><th>Fixed maps</th><th>Random maps</th></tr>')
    me = 'aljaz2_final'
    for o in sorted((o for o in order if o != me), key=lambda o: sum(h2h[(me, o)]) / max(1, len(h2h[(me, o)]))):
        p = h2h[(me, o)]
        fx = [res[(me, o, lay, red)] for lay in layouts if not lay.startswith('RANDOM') for red in (True, False)
              if (me, o, lay, red) in res]
        rn = [res[(me, o, lay, red)] for lay in layouts if lay.startswith('RANDOM') for red in (True, False)
              if (me, o, lay, red) in res]
        ppg = sum(p) / len(p)
        h.append('<tr><td>%s</td><td>%.0f%%</td>%s<td>%s</td><td>%s</td></tr>'
                 % (html.escape(label(o)), 100.0 * sum(1 for x in p if x == 3) / len(p), tag('%.2f' % ppg, cell_class(ppg)),
                    '%.2f' % (sum(fx) / len(fx)) if fx else '-', '%.2f' % (sum(rn) / len(rn)) if rn else '-'))
    h.append('</table>')

    h.append('<h2>Data checks</h2><ul>')
    h.append('<li>Maps: %s.</li>' % html.escape(', '.join(layouts)))
    h.append('<li>Runner errors (left out): %d.</li>' % len(errors))
    h.append('<li>Machine stalls (an agent time above 200 s in one game): %d.</li>' % faults[('machine', 'stall')])
    h.append('<li>Time warnings over all games: %d.</li>' % faults[('all', 'time warnings')])
    crash = [(t, c) for (t, kind), c in faults.items() if kind == 'crash or timeout']
    h.append('<li>Games lost to a crash or timeout: %s.</li>'
             % (', '.join('%s %d' % (html.escape(label(t)), c) for t, c in sorted(crash, key=lambda x: -x[1])) or 'none'))
    h.append('</ul><p class="muted">Time-bounded teams (UCT, MCTS, expectimax, hybrid) search less when the machine is '
             'busy; every game here ran under the same load, so the comparison is fair, but alone on a machine those '
             'teams may play somewhat stronger.</p></main></body></html>')
    open(out, 'w').write('\n'.join(h))
    print('games %d, errors %d, layouts %d, teams %d -> %s' % (len(rows), len(errors), len(layouts), len(teams), out))


STYLE = """<style>
:root { --bg: #fbfaf8; --fg: #1d1d1b; --muted: #6b6a66; --line: #e3e0da; --card: #ffffff;
  --good: #dff1e3; --mid: #f6efd9; --bad: #f6dfdc; --up: #2f7d4a; --down: #a23b2e; --ours: #eef3fb; }
@media (prefers-color-scheme: dark) { :root { --bg: #171716; --fg: #ecebe7; --muted: #a29f97; --line: #34332f;
  --card: #1f1f1d; --good: #1e3a27; --mid: #3a331d; --bad: #43231f; --up: #7fd19a; --down: #f09a8c; --ours: #1f2a3a; } }
body { background: var(--bg); color: var(--fg); font: 15px/1.5 -apple-system, "Segoe UI", sans-serif; margin: 0; }
main { max-width: 1200px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 26px; margin: 0 0 4px; } h2 { margin-top: 40px; border-bottom: 1px solid var(--line); }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; background: var(--card); }
th, td { border: 1px solid var(--line); padding: 6px 8px; text-align: left; vertical-align: top; }
table.res td { white-space: nowrap; } table.res td span { color: var(--muted); margin-left: 6px; font-size: 13px; }
table.mat th, table.mat td { padding: 4px 6px; font-size: 13px; text-align: center; } table.mat th:first-child { text-align: left; white-space: nowrap; }
td.good { background: var(--good); } td.mid { background: var(--mid); } td.bad { background: var(--bad); }
td.up { color: var(--up); font-weight: 600; } td.down { color: var(--down); font-weight: 600; }
tr.ours td { background: var(--ours); }
.muted { color: var(--muted); }
</style>"""

if __name__ == '__main__':
    main()
