"""Generate Markdown and standalone HTML from a frozen tournament run."""
import argparse
import html
import math
import statistics
from collections import Counter
from pathlib import Path

from evaluation.core import read_json


def empty_stats():
    return dict(wins=0, ties=0, losses=0, errors=0, forfeits=0, warnings=0, timeouts=0,
                crashes=0, scores=[], games=0, points=0)


def build_stats(manifest, rows):
    jobs = {j['match_id']: j for j in manifest['schedule']}
    teams = {t: empty_stats() for t in manifest['teams']}
    matrix, breakdown, seen = {}, {}, set()
    for r in rows:
        mid = r['match_id']
        if mid in seen or mid not in jobs:
            raise ValueError('Duplicate or unknown match result: ' + str(mid))
        if any(r.get(k) != v for k, v in jobs[mid].items()):
            raise ValueError('Result does not match frozen schedule: ' + str(mid))
        if r.get('status') not in ('completed', 'forfeit', 'error'):
            raise ValueError('Unknown match status')
        if r['status'] != 'error' and (type(r.get('score')) not in (int, float) or not math.isfinite(r['score'])):
            raise ValueError('Completed match needs a finite score')
        seen.add(mid)
        for side, other in [('red', 'blue'), ('blue', 'red')]:
            tid = r[side]
            groups = [teams[tid], matrix.setdefault((tid, r[other]), empty_stats()),
                      breakdown.setdefault((tid, 'colour', side), empty_stats()),
                      breakdown.setdefault((tid, 'layout', r['layout']), empty_stats())]
            for g in groups:
                for event in ('warnings', 'timeouts', 'crashes'):
                    g[event] += r.get('events', {}).get(side, {}).get(event, 0)
                if r['status'] == 'error':
                    g['errors'] += 1
                    continue
                score = r['score'] * (1 if side == 'red' else -1)
                outcome = 'wins' if score > 0 else 'losses' if score < 0 else 'ties'
                g[outcome] += 1
                g['games'] += 1
                g['points'] += 3 if score > 0 else 1 if score == 0 else 0
                g['scores'].append(score)
                if r['status'] == 'forfeit' and r.get('events', {}).get(side, {}).get('crashes', 0):
                    g['forfeits'] += 1
    return dict(teams=teams, matrix=matrix, breakdown=breakdown,
                pending=len(jobs) - len(seen), recorded=len(seen),
                incomplete_pairs=sum(n == 1 for n in Counter(jobs[mid]['pair_id'] for mid in seen).values()))


def interval(wins, total):
    if not total:
        return 'n/a'
    z = 1.96
    p = wins / total
    den = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / den
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return '{:.0%}-{:.0%}'.format(centre - radius, centre + radius)


def stat_row(t, s):
    scores = s['scores']
    return [t, s['games'], s['points'], '{}/{}/{}'.format(s['wins'], s['ties'], s['losses']),
            '{:.1%}'.format(s['wins'] / s['games']) if s['games'] else 'n/a', interval(s['wins'], s['games']),
            '{:.2f}'.format(statistics.mean(scores)) if scores else 'n/a',
            str(statistics.median(scores)) if scores else 'n/a',
            '{}..{}'.format(min(scores), max(scores)) if scores else 'n/a',
            s['errors'], s['forfeits'], s['warnings'], s['timeouts']]


HEADERS = ['Team', 'Games', 'Points', 'W/T/L', 'Win %', 'Approx. 95% interval', 'Mean score',
           'Median', 'Range', 'Errors', 'Own forfeits', 'Warnings', 'Timeout events']


def tables(manifest, stats):
    ranking = sorted(stats['teams'], key=lambda t: (-stats['teams'][t]['points'], t))
    league = [stat_row(t, stats['teams'][t]) for t in ranking]
    ids = sorted(stats['teams'])
    matrix = []
    for a in ids:
        cells = []
        for b in ids:
            s = stats['matrix'].get((a, b))
            cells.append('{}/{}/{}'.format(s['wins'], s['ties'], s['losses']) if s else '-')
        matrix.append([a] + cells)
    details = [[t, kind, value] + stat_row(t, s)[1:]
               for (t, kind, value), s in sorted(stats['breakdown'].items())]
    return [(HEADERS, league), (['Team vs opponent'] + ids, matrix),
            (['Team', 'Group', 'Value'] + HEADERS[1:], details)]


def md(value):
    return html.escape(str(value)).replace('|', '&#124;').replace('\n', ' ').replace('`', '&#96;')


def md_table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(map(md, headers)) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(md, row)) + ' |' for row in rows])


def report_note(manifest, stats):
    state = 'Partial' if stats['pending'] else 'Complete'
    short = manifest['config'].get('moves', 1200) < 1200
    return '{}: {}/{} matches recorded; {} pending; {} colour pairs incomplete.{} Workers: {}. Engine: {}.'.format(
        state, stats['recorded'], len(manifest['schedule']), stats['pending'], stats['incomplete_pairs'],
        ' SHORT GAME VALIDATION ONLY, not a strength comparison.' if short else '',
        ','.join(map(str, sorted({session['workers'] for session in manifest.get('sessions', [])}))) or str(manifest['workers']), manifest['engine_hash'][:12])


def render_markdown(manifest, rows):
    stats = build_stats(manifest, rows)
    parts = ['# Tournament ' + md(manifest['run_id']), report_note(manifest, stats),
             'League points: win 3, tie 1, loss 0. Forfeits count with the engine outcome; runner errors do not. '
             'Compare coverage before ranking a partial run. Intervals describe win proportion, treat ties as non-wins, '
             'and assume independent games; repeated maps may violate this assumption. '
             'Parallel timing describes a loaded machine and does not prove standalone timing safety.']
    for title, (headers, data) in zip(['Ranking', 'Pairwise W/T/L (row perspective)', 'Colour and layout breakdown'], tables(manifest, stats)):
        parts.extend(['## ' + title, md_table(headers, data)])
    parts.append('## Teams and frozen versions')
    for tid, t in sorted(manifest['teams'].items()):
        parts.append('- **{}**: {}. Author: {}. Approach: {}. Components: {}. Source SHA256: {}.'.format(
            md(tid), md(t['description']), md(t['author']), md(t.get('approach', '')), md(', '.join(t.get('components', []))), md(t['hash'])))
    parts.extend(['## Match records', md_table(['Match', 'Red', 'Blue', 'Layout', 'Seed', 'Status', 'Red score', 'Seconds', 'Diagnostic'],
        [[r['match_id'], r['red'], r['blue'], r['layout'], r['seed'], r['status'], r['score'], r['elapsed'], r.get('error') or ''] for r in sorted(rows, key=lambda r: r['match_id'])]),
        'Exact configuration, versions and schedule: `manifest.json`. Raw results: `matches.jsonl`. '
        'Detailed logs and optional replays remain under the local run directory; they are not included in this shared report.'])
    return '\n\n'.join(parts) + '\n'


def html_table(headers, rows):
    esc = lambda v: html.escape(str(v), quote=True)
    return '<div class="scroll"><table><thead><tr>' + ''.join('<th>' + esc(v) + '</th>' for v in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join('<td>' + esc(v) + '</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def render_html(manifest, rows):
    stats = build_stats(manifest, rows)
    esc = lambda v: html.escape(str(v), quote=True)
    body = '<h1>Tournament ' + esc(manifest['run_id']) + '</h1><p>' + esc(report_note(manifest, stats)) + '</p>'
    body += '<p>Points: 3/1/0. Errors are not ties. Forfeits retain the engine result. Compare coverage in partial runs. Approximate win intervals assume independent games. Validate finalist timing separately.</p>'
    for title, (headers, data) in zip(['Ranking', 'Pairwise W/T/L', 'Colour and layout breakdown'], tables(manifest, stats)):
        body += '<h2>' + title + '</h2>' + html_table(headers, data)
    body += '<h2>Frozen teams</h2>'
    for tid, t in sorted(manifest['teams'].items()):
        body += '<h3>' + esc(tid) + '</h3><p>' + esc(t['description']) + ' | ' + esc(t['author']) + ' | ' + esc(t.get('approach', '')) + '</p><p>Source SHA256: <code>' + esc(t['hash']) + '</code></p>'
    body += '<h2>Match explorer</h2><div class="filters">'
    for key, values in [('team', sorted(manifest['teams'])), ('layout', sorted({j['layout'] for j in manifest['schedule']})), ('status', ['completed', 'forfeit', 'error'])]:
        body += '<label>' + key.capitalize() + ' <select id="' + key + '"><option value="">All</option>' + ''.join('<option value="' + esc(v) + '">' + esc(v) + '</option>' for v in values) + '</select></label>'
    body += '</div><p id="count"></p><div class="scroll"><table><thead><tr>' + ''.join('<th>' + v + '</th>' for v in ['Match', 'Red', 'Blue', 'Layout', 'Seed', 'Status', 'Red score', 'Seconds', 'Diagnostic']) + '</tr></thead><tbody id="matches">'
    for r in sorted(rows, key=lambda r: r['match_id']):
        body += '<tr ' + ' '.join('data-' + k + '="' + esc(r[k]) + '"' for k in ['red', 'blue', 'layout', 'status']) + '>' + ''.join('<td>' + esc(r[k]) + '</td>' for k in ['match_id', 'red', 'blue', 'layout', 'seed', 'status', 'score', 'elapsed']) + '<td>' + esc(r.get('error') or '') + '</td></tr>'
    body += '</tbody></table></div>'
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PacLab tournament</title>
<style>body{font:15px system-ui,sans-serif;background:#f4f5f3;color:#182b28;margin:0;padding:32px;max-width:1500px}h1{font-size:32px}h2{margin-top:40px;border-top:2px solid #263f38;padding-top:16px}p{line-height:1.6}.scroll{overflow:auto}table{border-collapse:collapse;background:white;width:100%;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:10px;border-bottom:1px solid #d6dfdb;white-space:nowrap}th{background:#dfe9e3}tr:hover{background:#eef4f0}.filters{display:flex;gap:24px;flex-wrap:wrap}select{font:inherit;padding:8px}code{overflow-wrap:anywhere}tr[hidden]{display:none}</style><body>''' + body + '''<script>
function filter(){const t=document.getElementById('team').value,l=document.getElementById('layout').value,s=document.getElementById('status').value;let n=0;document.querySelectorAll('#matches tr').forEach(r=>{r.hidden=!!((t&&r.dataset.red!==t&&r.dataset.blue!==t)||(l&&r.dataset.layout!==l)||(s&&r.dataset.status!==s));if(!r.hidden)n++});document.getElementById('count').textContent=n+' matches shown'}document.querySelectorAll('select').forEach(s=>s.addEventListener('change',filter));filter();
</script></body></html>'''


def load_rows(path):
    if not path.exists():
        return []
    return [__import__('json').loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def build_report(run_dir):
    run_dir = Path(run_dir)
    manifest = read_json(run_dir / 'manifest.json')
    rows = load_rows(run_dir / 'matches.jsonl')
    (run_dir / 'report.md').write_text(render_markdown(manifest, rows), encoding='utf-8')
    (run_dir / 'report.html').write_text(render_html(manifest, rows), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='reports/<run-id> directory')
    args = parser.parse_args()
    build_report(args.run)


if __name__ == '__main__':
    main()
