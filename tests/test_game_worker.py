import contextlib
import io
import random
import unittest
from unittest.mock import patch
from evaluation.game_worker import execute


class SeedTests(unittest.TestCase):
    def test_seed_reset_occurs_after_layout_generation(self):
        observed = []
        def parse(args):
            random.seed(7)  # mimic RANDOM7 overriding the requested game seed
            return dict(red_team_name='Red', blue_team_name='Blue')
        def games(**kwargs):
            observed.append(random.random())
            return []
        with patch('evaluation.game_worker.capture.read_command', side_effect=parse), patch('evaluation.game_worker.capture.run_games', side_effect=games), contextlib.redirect_stdout(io.StringIO()):
            execute(['--setRandomSeed', '17'])
        self.assertEqual(observed, [random.Random('17').random()])
