"""Real engine smoke tests: snapshots, concurrent games and graceful stop/resume."""
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class IntegrationTests(unittest.TestCase):
    def test_real_games_soft_stop_resume_and_reports(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            shutil.copytree(str(ROOT / 'evaluation'), str(root / 'evaluation'))
            engine = root / 'pacman-contest/src/contest'
            engine.mkdir(parents=True)
            for p in (ROOT / 'pacman-contest/src/contest').glob('*.py'):
                shutil.copy2(str(p), str(engine / p.name))
            shutil.copytree(str(ROOT / 'pacman-contest/src/contest/layouts'), str(engine / 'layouts'))
            (root / 'my_team.py').write_text('''import time
from contest.baseline_team import OffensiveReflexAgent, DefensiveReflexAgent
class SlowOffense(OffensiveReflexAgent):
    def choose_action(self, state):
        time.sleep(0.03)
        return super().choose_action(state)
class SlowDefense(DefensiveReflexAgent):
    def choose_action(self, state):
        time.sleep(0.03)
        return super().choose_action(state)
def create_team(a, b, is_red):
    return [SlowOffense(a), SlowDefense(b)]
''')
            config = dict(teams=['submission', 'baseline'], layouts=['defaultCapture', 'RANDOM7', 'officeCapture'],
                          repeats=1, seed=7, moves=40, record=False)
            (root / 'suite.json').write_text(json.dumps(config))
            command = [sys.executable, '-B', '-m', 'evaluation.run_tournament']
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
            env.pop('PYTHONPATH', None)
            output = root / 'runner.log'
            with output.open('w') as out:
                proc = subprocess.Popen(command + ['--config', 'suite.json', '--jobs', '2', '--run-id', 'integration'],
                                        cwd=str(root), env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    deadline = time.monotonic() + 15
                    while len(list((root / 'local-runs/integration/matches').glob('*/game.log'))) < 2:
                        if proc.poll() is not None or time.monotonic() > deadline:
                            self.fail('Runner did not start games: ' + output.read_text())
                        time.sleep(0.02)
                    os.killpg(proc.pid, signal.SIGINT)  # terminal-group interrupt, not just parent PID
                    self.assertEqual(proc.wait(timeout=20), 0, output.read_text())
                finally:
                    if proc.poll() is None:
                        proc.kill()
                        proc.wait()
            report = root / 'reports/integration'
            rows = [json.loads(x) for x in (report / 'matches.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 2, output.read_text())
            self.assertTrue(all(r['status'] == 'completed' for r in rows), [p.read_text() for p in (root / 'local-runs/integration/matches').glob('*/game.log')])
            self.assertIn('Partial', (report / 'report.md').read_text())
            # Resume must use the snapshot even if the live submission is now broken.
            (root / 'my_team.py').write_text('raise RuntimeError("live source changed")')
            resume = subprocess.run(command + ['--resume', 'integration', '--jobs', '2'], cwd=str(root), env=env,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(resume.returncode, 0, resume.stdout + resume.stderr)
            all_rows = [json.loads(x) for x in (report / 'matches.jsonl').read_text().splitlines()]
            self.assertEqual(len(all_rows), 6)
            self.assertEqual(len({r['match_id'] for r in all_rows}), 6)
            self.assertTrue(all(r['status'] == 'completed' for r in all_rows), all_rows)
            self.assertIn('Complete', (report / 'report.md').read_text())
            self.assertIn('SHORT GAME VALIDATION ONLY', (report / 'report.md').read_text())
            self.assertIn('Match explorer', (report / 'report.html').read_text())
            self.assertFalse((root / 'local-runs/integration/runner.lock').exists())
            again = subprocess.run(command + ['--resume', 'integration'], cwd=str(root), env=env,
                                   capture_output=True, text=True, timeout=15)
            self.assertEqual(again.returncode, 0, again.stderr)
            self.assertEqual((report / 'matches.jsonl').read_text().splitlines(),
                             [json.dumps(r, sort_keys=True) for r in all_rows])
            regenerated = subprocess.run([sys.executable, '-B', '-m', 'evaluation.build_report', 'reports/integration'],
                                         cwd=str(root), env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual(regenerated.returncode, 0, regenerated.stderr)
