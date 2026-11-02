"""ASCII frames of a recorded game.

    venv/bin/python -B experiments/tools/show.py OUT.jsonl ROW_INDEX FROM_STEP TO_STEP [EVERY]
Agents are drawn as digits 0-3 (upper-case P marker in the legend when pacman); food '.', capsule 'o'.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from match import ACTION, build_layout  # noqa: E402
from contest import capture  # noqa: E402
import contest.text_display as text_display  # noqa: E402


def frame(state):
    walls = state.get_walls()
    food = state.data.food
    caps = set(state.get_capsules())
    pos = {tuple(int(v) for v in state.get_agent_state(i).get_position()): str(i) for i in range(4)}
    lines = []
    for y in range(walls.height - 1, -1, -1):
        row = ''
        for x in range(walls.width):
            if (x, y) in pos:
                row += pos[(x, y)]
            elif walls[x][y]:
                row += '#'
            elif (x, y) in caps:
                row += 'o'
            elif food[x][y]:
                row += '.'
            else:
                row += ' '
        lines.append(row)
    return '\n'.join(lines)


def main():
    rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    row = rows[int(sys.argv[2])]
    lo, hi = int(sys.argv[3]), int(sys.argv[4])
    every = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    rules = capture.CaptureRules()
    rules.quiet = True
    state = rules.new_game(build_layout(row['layout']), [None] * 4, text_display.NullGraphics(),
                           1200, False, False).state
    print(row['red'], 'vs', row['blue'], row['layout'], 'final', row['score'])
    moves = row['moves']
    for step in range(len(moves) // 2):
        idx, a = int(moves[2 * step]), moves[2 * step + 1]
        state = state.generate_successor(idx, ACTION[a])
        if lo <= step <= hi and (step - lo) % every == 0:
            info = ' '.join(f"{i}:{'P' if s.is_pacman else 'g'}{s.num_carrying}"
                            f"{'s' + str(s.scared_timer) if s.scared_timer else ''}"
                            for i, s in enumerate(state.data.agent_states))
            print(f'--- step {step} score {state.data.score} | {info}')
            print(frame(state))


if __name__ == '__main__':
    main()
