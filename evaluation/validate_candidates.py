"""Reliability checks against the course baseline, not a candidate ranking league."""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from evaluation.core import ID, discover_teams, read_json
from evaluation.run_tournament import copy_tree, tree_hash
ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('teams', nargs='+')
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--moves', type=int, default=1200)
    parser.add_argument('--layouts', nargs='+', default=['defaultCapture', 'RANDOM7'])
    args = parser.parse_args()
    if not ID.fullmatch(args.prefix):
        parser.error('Invalid prefix')
    known = discover_teams(ROOT)
    for tid in args.teams:
        if tid not in known or tid == 'baseline':
            parser.error('Unknown candidate or baseline cannot play itself: ' + tid)
    failures = []
    for tid in args.teams:
        run_id = args.prefix + '-' + tid.replace('aljaz_', '').replace('control_', '')
        config_path = ROOT / 'local-runs' / 'validation-configs' / (run_id + '.json')
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(dict(teams=[tid, 'baseline'], layouts=args.layouts,
                                               repeats=1, seed=17, moves=args.moves, record=False)))
        command = [sys.executable, '-B', '-m', 'evaluation.run_tournament', '--jobs', str(args.jobs)]
        if (ROOT / 'reports' / run_id / 'manifest.json').exists():
            old = read_json(ROOT / 'reports' / run_id / 'manifest.json')
            with tempfile.TemporaryDirectory() as temporary:
                target = Path(temporary) / 'team'
                t = known[tid]
                if tid == 'submission':
                    target.mkdir()
                    import shutil
                    for source in ROOT.glob('*.py'):
                        shutil.copy2(str(source), str(target / source.name))
                else:
                    copy_tree(ROOT / t['folder'], target)
                if old['teams'][tid]['hash'] != tree_hash(target) or old['config']['moves'] != args.moves or old['config']['layouts'] != args.layouts:
                    parser.error('Candidate/config changed; use a new validation --prefix: ' + tid)
            command += ['--resume', run_id]
        else:
            command += ['--config', str(config_path), '--run-id', run_id]
        result = subprocess.run(command, cwd=str(ROOT), env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        path = ROOT / 'reports' / run_id / 'matches.jsonl'
        rows = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
        good = result.returncode == 0 and len(rows) == 2 * len(args.layouts) and all(
            r['status'] == 'completed' and not any(sum(e.values()) for e in r['events'].values()) for r in rows)
        print('{} reliability: {}'.format(tid, 'PASS' if good else 'FAIL'), flush=True)
        if not good:
            failures.append(tid)
    if failures:
        parser.exit(1, 'Failed reliability checks: ' + ', '.join(failures) + '\n')


if __name__ == '__main__':
    main()
