"""Replay recorded games and check that a team picks every recorded move of its own (no new games are played).

    venv/bin/python -B experiments/tools/replay_check.py TEAM_DIR RESULTS.jsonl ['{"param": 1}'] [EVERY]
Our agents are re-created on the recorded layout and asked for a move at every recorded state; the opponent's
recorded moves are applied as played. 0 mismatches means the code plays those games exactly as recorded.
EVERY checks only every n-th game.
"""
import collections
import contextlib
import io
import json
import os
import sys

os.environ['PACLAB_PARAMS'] = sys.argv[3] if len(sys.argv) > 3 else '{}'
sys.path.insert(0, os.path.abspath(sys.argv[1]))
from contest import capture
import contest.text_display as text_display
from match import ACTION, build_layout
import my_team


def main():
    rows = [json.loads(line) for line in open(sys.argv[2])]
    every = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    me = (collections.Counter(r['red'] for r in rows) + collections.Counter(r['blue'] for r in rows)).most_common(1)[0][0]
    games = moves = bad = 0
    for k, r in enumerate(rows):
        if k % every:
            continue
        red = r['red'] == me
        ours = [0, 2] if red else [1, 3]
        with contextlib.redirect_stdout(io.StringIO()):
            rules = capture.CaptureRules()
            rules.quiet = True
            state = rules.new_game(build_layout(r['layout']), [None] * 4, text_display.NullGraphics(),
                                   1200, False, False).state
            agents = my_team.create_team(ours[0], ours[1], red)
            for a in agents:
                a.register_initial_state(state)
        mv = r['moves']
        for step in range(len(mv) // 2):
            idx, act = int(mv[2 * step]), ACTION[mv[2 * step + 1]]
            if idx in ours:
                moves += 1
                with contextlib.redirect_stdout(io.StringIO()):
                    chosen = agents[ours.index(idx)].choose_action(state)
                bad += chosen != act
            state = state.generate_successor(idx, act)
        games += 1
    print('games %d our moves %d mismatches %d' % (games, moves, bad))


if __name__ == '__main__':
    main()
