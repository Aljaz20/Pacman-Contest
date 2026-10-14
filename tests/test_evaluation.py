import json
import tempfile
import unittest
from pathlib import Path

from evaluation.core import schedule, classify, validate_config, discover_teams
from evaluation.build_report import build_stats, render_markdown, render_html


class CoreTests(unittest.TestCase):
    def test_colour_pairs_share_seed_and_cover_all_opponents(self):
        jobs = schedule(['a', 'b', 'c'], ['defaultCapture', 'RANDOM7'], 2, 11)
        self.assertEqual(len(jobs), 24)
        self.assertEqual(len({j['match_id'] for j in jobs}), 24)
        for i in range(0, len(jobs), 2):
            a, b = jobs[i:i + 2]
            self.assertEqual((a['red'], a['blue']), (b['blue'], b['red']))
            self.assertEqual(a['seed'], b['seed'])
            self.assertEqual(a['pair_id'], b['pair_id'])
        self.assertEqual(jobs, schedule(['a', 'b', 'c'], ['defaultCapture', 'RANDOM7'], 2, 11))

    def test_config_rejects_duplicates_and_invalid_counts(self):
        for change in ({'teams': ['a', 'a']}, {'repeats': 0}, {'moves': True}, {'layouts': []}):
            c = dict(teams=['a', 'b'], layouts=['defaultCapture'], repeats=1, seed=0, moves=1200)
            c.update(change)
            with self.assertRaises(ValueError):
                validate_config(c)

    def test_missing_score_is_error_not_tie(self):
        self.assertEqual(classify(None, '', 0)['status'], 'error')
        self.assertIsNone(classify(None, '', 0)['winner'])
        self.assertEqual(classify(0, '', 0)['status'], 'completed')

    def test_forfeit_is_distinct_and_has_engine_winner(self):
        r = classify(-1, 'Agent 0 timed out on a single move!\nRed agent crashed', 0)
        self.assertEqual(r['status'], 'forfeit')
        self.assertEqual(r['winner'], 'blue')
        self.assertEqual(r['events']['red']['timeouts'], 1)
        self.assertEqual(r['events']['red']['crashes'], 1)
        self.assertEqual(classify(4, 'fatal', 1)['status'], 'error')

    def test_nonfinite_score_is_error(self):
        self.assertEqual(classify(float('nan'), '', 0)['status'], 'error')

    def test_team_entry_cannot_escape_folder(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'teams' / 'evil'
            p.mkdir(parents=True)
            (p / 'team.json').write_text(json.dumps(dict(id='evil', author='a', description='b',
                                                        approach='c', entry='../outside.py', options={})))
            with self.assertRaises(ValueError):
                discover_teams(Path(d))


class ReportTests(unittest.TestCase):
    def manifest(self):
        return {'run_id': 'test', 'teams': {'a': {'author': 'A', 'description': '<script>x</script>', 'hash': '123'},
                                         'b': {'author': 'B', 'description': 'test', 'hash': '456'}},
                'schedule': schedule(['a', 'b'], ['defaultCapture'], 1, 1),
                'config': {'moves': 1200}, 'engine_hash': '789', 'workers': 2}

    def test_partial_denominators_and_error_not_ranked(self):
        m = self.manifest()
        rows = [dict(m['schedule'][0], status='completed', score=3, winner='red', elapsed=1,
                     events={'red': {}, 'blue': {}})]
        stats = build_stats(m, rows)
        self.assertEqual(stats['pending'], 1)
        self.assertEqual(stats['teams']['a']['wins'], 1)
        self.assertEqual(stats['teams']['b']['losses'], 1)
        html = render_html(m, rows)
        self.assertNotIn('<script>x</script>', html)
        self.assertIn('&lt;script&gt;x&lt;/script&gt;', html)
        self.assertIn('Partial', render_markdown(m, rows))

    def test_duplicate_results_rejected(self):
        m = self.manifest()
        row = dict(m['schedule'][0], status='error', score=None, winner=None, elapsed=1,
                   events={'red': {}, 'blue': {}})
        with self.assertRaises(ValueError):
            build_stats(m, [row, row])


class RunnerTests(unittest.TestCase):
    def test_snapshot_rejects_symlinks(self):
        from evaluation.run_tournament import copy_tree
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / 'src').mkdir()
            (p / 'outside').write_text('secret')
            (p / 'src' / 'asset').symlink_to(p / 'outside')
            with self.assertRaises(ValueError):
                copy_tree(p / 'src', p / 'dst')

    def test_resume_rejects_changed_snapshot(self):
        from evaluation.run_tournament import tree_hash, verify_snapshots
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / 'engine').mkdir()
            (p / 'engine' / 'x.py').write_text('x=1')
            (p / 'teams' / 'a').mkdir(parents=True)
            (p / 'teams' / 'a' / 'my_team.py').write_text('x=1')
            m = {'engine_hash': tree_hash(p / 'engine'),
                 'teams': {'a': {'hash': tree_hash(p / 'teams' / 'a')}}}
            verify_snapshots(p, m)
            (p / 'teams' / 'a' / 'my_team.py').write_text('x=2')
            with self.assertRaises(ValueError):
                verify_snapshots(p, m)


class MetadataTests(unittest.TestCase):
    def test_components_are_a_list_of_strings(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'teams' / 'valid'
            p.mkdir(parents=True)
            (p / 'my_team.py').write_text('pass')
            (p / 'team.json').write_text(json.dumps(dict(id='valid', author='a', description='b',
                       approach='c', entry='my_team.py', options={}, components=17)))
            with self.assertRaises(ValueError):
                discover_teams(Path(d))


class HelperTests(unittest.TestCase):
    def test_conflicting_helper_modules_are_rejected(self):
        from evaluation.run_tournament import check_helpers
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in ['a', 'b']:
                folder = root / name
                folder.mkdir()
                (folder / 'my_team.py').write_text('pass')
                (folder / 'shared.py').write_text('value=' + repr(name))
            with self.assertRaises(ValueError):
                check_helpers(root, {'a': {'entry': 'my_team.py'}, 'b': {'entry': 'my_team.py'}})
