"""Build experiments/report.html: experiment steps from LOG.md plus per-opponent tables of each version.

    venv/bin/python -B experiments/tools/report.py experiments/report.json experiments/report.html
report.json (results paths relative to it): {"columns": ["Codex v27", "v1a", ...], "diffs": [["v1a", "final"], ...],
              "tables": {"Holdout (30 unseen maps)": {"Codex v27": "path.jsonl", ...}, ...}}
Each results file holds one team (the column) against the report pool, both colours.
"""
import collections
import html
import json
import os
import re
import sys

GROUPS = [('Simple teams', ['baseline_team', 'control_double_defense', 'control_greedy_food']),
          ('Hard teams (tuning pool)', ['l2_team', 'aljaz_expectimax_v37_team_plan_b200', 'aljaz_hybrid_v42_turn_guard',
                                       'aljaz_tactical_uct_v7_b200', 'aljaz_astar_v27_full_state',
                                       'aljaz_astar_v26_pursuit', 'aljaz_mdp_frontier_v4']),
          ('Unseen teams (never tuned against)', ['aljaz_minimax_v3', 'aljaz_neural_v28_pursuit',
                                                  'aljaz_route_uct_v12_turn_guard_b200'])]
NAMES = {'baseline_team': 'Course baseline', 'control_double_defense': 'Double defence (border campers)',
         'control_greedy_food': 'Greedy food eater', 'l2_team': 'L2 (private)',
         'aljaz_expectimax_v37_team_plan_b200': 'Codex Expectimax v37', 'aljaz_hybrid_v42_turn_guard': 'Codex Hybrid v42',
         'aljaz_tactical_uct_v7_b200': 'Codex Tactical UCT v7', 'aljaz_astar_v27_full_state': 'Codex A-Star v27',
         'aljaz_astar_v26_pursuit': 'Codex A-Star v26', 'aljaz_mdp_frontier_v4': 'Codex MDP v4',
         'aljaz_minimax_v3': 'Codex Minimax v3', 'aljaz_neural_v28_pursuit': 'Codex Neural v28',
         'aljaz_route_uct_v12_turn_guard_b200': 'Codex Route UCT v12'}


def name_of(key):
    """Readable opponent name; teams outside NAMES get one from their folder name."""
    if key in NAMES:
        return NAMES[key]
    if key.startswith('aljaz_'):
        return 'Codex ' + key[6:].replace('_', ' ')
    return key.replace('_', ' ')


def key_of(path):
    base = os.path.basename(path)
    return base[:-3] if base != 'my_team.py' else os.path.basename(os.path.dirname(path))


def load(path):
    """{opponent key: [points per game]} for the team that appears in every row."""
    rows = [json.loads(line) for line in open(path)]
    count = collections.Counter(r['red'] for r in rows) + collections.Counter(r['blue'] for r in rows)
    me = count.most_common(1)[0][0]
    out = collections.defaultdict(list)
    for r in rows:
        red = r['red'] == me
        opp = key_of(r['blue'] if red else r['red'])
        if opp == key_of(me):
            continue  # mirror match
        sc = r['score'] if red else -r['score']
        out[opp].append(3 if sc > 0 else 1 if sc == 0 else 0)
    return out


def cell(pts):
    if not pts:
        return '<td class="na">-</td>'
    win = 100.0 * sum(p == 3 for p in pts) / len(pts)
    ppg = sum(pts) / len(pts)
    hue = 'good' if ppg >= 2.4 else 'mid' if ppg >= 1.8 else 'bad'
    return '<td class="%s"><b>%.0f%%</b><span>%.2f</span></td>' % (hue, win, ppg)


def diff_cell(new, old):
    if not new or not old:
        return '<td></td>'
    d = sum(new) / len(new) - sum(old) / len(old)
    return '<td class="%s">%+.2f</td>' % ('up' if d > 0 else 'down' if d < 0 else '', d)


def table(title, files, columns, diffs):
    data = {c: load(files[c]) for c in columns if c in files and os.path.exists(files[c])}
    cols = [c for c in columns if c in data]
    diffs = [(a, b) for a, b in diffs if a in data and b in data]
    out = ['<h3>%s</h3>' % html.escape(title), '<div class="scroll"><table class="res"><tr><th>Opponent</th>']
    out += ['<th>%s</th>' % html.escape(c) for c in cols]
    out += ['<th>%s &minus; %s<br><small>PPG</small></th>' % (html.escape(b), html.escape(a)) for a, b in diffs]
    out.append('</tr>')
    pooled = {c: [] for c in cols}
    known = {k for _, keys in GROUPS for k in keys}
    other = sorted({k for c in cols for k in data[c]} - known)
    for group, keys in GROUPS + [('Other teams (never tuned against)', other)]:
        if not any(k in data[c] for c in cols for k in keys):
            continue
        out.append('<tr class="grp"><td colspan="%d">%s</td></tr>' % (len(cols) + len(diffs) + 1,
                                                                       html.escape(group)))
        for k in keys:
            if not any(k in data[c] for c in cols):
                continue
            out.append('<tr><td>%s</td>' % html.escape(name_of(k)))
            for c in cols:
                pooled[c] += data[c].get(k, [])
                out.append(cell(data[c].get(k, [])))
            out += [diff_cell(data[b].get(k), data[a].get(k)) for a, b in diffs]
            out.append('</tr>')
    out.append('<tr class="all"><td>All opponents</td>' + ''.join(cell(pooled[c]) for c in cols))
    out += [diff_cell(pooled[b], pooled[a]) for a, b in diffs]
    out.append('</tr></table></div>')
    return '\n'.join(out)


def log_rows(path):
    rows = []
    for line in open(path):
        m = re.match(r'^\| (\d+) \| (.*) \| (.*) \| (.*) \|\s*$', line)
        if m:
            rows.append(m.groups())
    return rows


def verdict_class(v):
    v = v.lower()
    if v.startswith('kept') or v.startswith('fixed') or 'enabled' in v or v.startswith('adopted'):
        return 'kept'
    if v.startswith('rejected') or v.startswith('bug'):
        return 'rej'
    return 'neu'


def inline(text):
    text = html.escape(text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
    return re.sub(r'`([^`]+)`', r'<code>\1</code>', text)


def main():
    cfg = json.load(open(sys.argv[1]))
    here = os.path.dirname(os.path.abspath(sys.argv[1]))
    columns = cfg['columns']
    parts = []
    for title, files in cfg['tables'].items():
        files = {c: os.path.join(here, f) for c, f in files.items()}  # paths are relative to report.json
        parts.append(table(title, files, columns, cfg.get('diffs', [])))
    steps = ['<tr class="%s"><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>' % (
        verdict_class(v), n, inline(c), inline(r), inline(v)) for n, c, r, v in log_rows(os.path.join(here, 'LOG.md'))]
    notes = ''.join('<li>%s</li>' % inline(n) for n in cfg.get('notes', []))
    page = TEMPLATE.format(summary=''.join('<p>%s</p>' % inline(p) for p in cfg.get('summary', [])),
                           tables='\n'.join(parts), steps='\n'.join(steps), notes=notes)
    open(sys.argv[2], 'w').write(page)


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>aljaz2 Pacman report</title>
<style>
:root {{ --bg: #fbfaf8; --fg: #1d1d1b; --muted: #6b6a66; --line: #e3e0da; --card: #ffffff;
  --good: #dff1e3; --mid: #f6efd9; --bad: #f6dfdc; --kept: #2f7d4a; --rej: #a23b2e; --neu: #7a7468; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg: #171716; --fg: #ecebe7; --muted: #a29f97; --line: #34332f;
  --card: #1f1f1d; --good: #1e3a27; --mid: #3a331d; --bad: #43231f; --kept: #7fd19a; --rej: #f09a8c; --neu: #b3ada2; }} }}
body {{ background: var(--bg); color: var(--fg); font: 15px/1.5 -apple-system, "Segoe UI", sans-serif; margin: 0; }}
main {{ max-width: 1100px; margin: 0 auto; padding: 24px 16px 64px; }}
h1 {{ font-size: 26px; margin: 0 0 4px; }} h2 {{ margin-top: 40px; border-bottom: 1px solid var(--line); }}
.scroll {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; background: var(--card); }}
th, td {{ border: 1px solid var(--line); padding: 6px 8px; text-align: left; vertical-align: top; }}
table.res td {{ white-space: nowrap; }} table.res td span {{ color: var(--muted); margin-left: 6px; font-size: 13px; }}
td.good {{ background: var(--good); }} td.mid {{ background: var(--mid); }} td.bad {{ background: var(--bad); }}
td.up {{ color: var(--kept); font-weight: 600; }} td.down {{ color: var(--rej); font-weight: 600; }}
tr.grp td {{ font-weight: 600; color: var(--muted); background: transparent; }}
tr.all td {{ font-weight: 700; }}
table.steps td:first-child {{ width: 32px; color: var(--muted); }}
tr.kept td:last-child {{ color: var(--kept); }} tr.rej td:last-child {{ color: var(--rej); }}
tr.neu td:last-child {{ color: var(--neu); }}
code {{ font-size: 13px; }} .muted {{ color: var(--muted); }}
</style></head><body><main>
<h1>aljaz2 Pacman agent &mdash; experiment report</h1>
<p class="muted">Every number is from games under tournament limits (Python 3.8, <code>-q -c</code>), both colours.
Cells: win % and average points per game (win 3, draw 1, loss 0).</p>
{summary}
<h2>Results by version</h2>
{tables}
<h2>Notes</h2><ul>{notes}</ul>
<h2>Every step</h2>
<p class="muted">From experiments/LOG.md. Green = kept, red = rejected, grey = neutral / kept off.</p>
<div class="scroll"><table class="steps"><tr><th>#</th><th>Change</th><th>Result</th><th>Verdict and reason</th></tr>
{steps}
</table></div>
</main></body></html>
"""

if __name__ == '__main__':
    main()
