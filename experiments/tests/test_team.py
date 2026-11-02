"""Unit tests for the aljaz2 team (run: venv/bin/python -B -m unittest discover -s experiments/tests)."""
import importlib.util
import os
import unittest

from contest import capture, layout as layout_mod
import contest.text_display as text_display

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEAM = os.environ.get('PACLAB_TEAM', os.path.join(ROOT, 'teams', 'paclab', 'my_team.py'))


def load_team():
    spec = importlib.util.spec_from_file_location('aljaz2_team_under_test', TEAM)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_state(rows, agents):
    lay = layout_mod.Layout(layout_name='t', layout_text=rows)
    rules = capture.CaptureRules()
    rules.quiet = True
    game = rules.new_game(lay, agents, text_display.NullGraphics(), 1200, False, False)
    return game.state


class MazeTest(unittest.TestCase):
    def setUp(self):
        self.mod = load_team()

    def test_distances_symmetric_and_borders(self):
        rows = ['%%%%%%%%', '%1 .. 2%', '%3 .. 4%', '%%%%%%%%']
        team = self.mod.create_team(0, 2, True)
        state = make_state(rows, [team[0], None, team[1], None])
        maze = self.mod.Maze(state.get_walls(), True)
        for a in maze.cells:
            for b in maze.cells:
                self.assertEqual(maze.d(a, b), maze.d(b, a))
        self.assertTrue(all(c[0] == 3 for c in maze.home_border))
        self.assertTrue(all(c[0] == 4 for c in maze.enemy_border))


class FieldTest(unittest.TestCase):
    """The escape field must forbid cells a ghost can reach long before we can get out again."""

    def setUp(self):
        self.mod = load_team()

    def test_pocket_is_unsafe_when_ghost_guards_exit(self):
        rows = [
            '%%%%%%%%%%%%',
            '%1....%....%',
            '%3....%.%%.%',
            '%..........%',
            '%.....%%%%2%',
            '%.....%%%%4%',
            '%%%%%%%%%%%%',
        ]
        red = self.mod.create_team(0, 2, True)
        blue = self.mod.create_team(1, 3, False)
        state = make_state(rows, [red[0], blue[0], red[1], blue[1]])
        agent = red[0]
        agent.register_initial_state(state)
        red[1].register_initial_state(state)
        s = agent.snapshot(state)
        T = agent.threat_field(s)
        L = agent.latest_field(s, T)
        # Engine y grows upward (text row r -> y = height-1-r): (10, 5) is the top-right corner cell.
        corner = (10, 5)
        self.assertIn(corner, L)
        # Blue ghost at (10, 2) is 3 steps away; our pacman at (1, 5) needs ~12: no safe visit.
        self.assertLess(L[corner], 0)
        # Cells right at the border stay enterable.
        self.assertGreaterEqual(L[(6, 3)], 0)


class RoleTest(unittest.TestCase):
    def setUp(self):
        self.mod = load_team()

    def test_roles_stable_for_repeated_identical_state(self):
        rows = ['%%%%%%%%%%', '%1......2%', '%3......4%', '%%%%%%%%%%']
        red = self.mod.create_team(0, 2, True)
        blue = self.mod.create_team(1, 3, False)
        state = make_state(rows, [red[0], blue[0], red[1], blue[1]])
        for a in red:
            a.register_initial_state(state)
        seen = []
        for k in range(6):
            agent = red[k % 2]
            state.data.timeleft = 1200 - k
            s = agent.snapshot(state)
            s['our_food_set'] = set(s['our_food'])
            roles = dict(agent.assign_roles(state, s))
            seen.append(roles)
        self.assertTrue(all(r == seen[0] for r in seen), seen)


if __name__ == '__main__':
    unittest.main()
