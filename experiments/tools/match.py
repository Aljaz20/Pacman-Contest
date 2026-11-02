"""Diagnostic match runner: plays one game and replays it to extract per-agent events.

Usage (single game, prints one JSON line):
    venv/bin/python -B experiments/tools/match.py RED_TEAM.py BLUE_TEAM.py LAYOUT SEED
LAYOUT is a fixed layout name (e.g. defaultCapture) or RANDOM<n>.
"""
import contextlib
import io
import json
import os
import random
import sys
import time

from contest import capture, layout as layout_mod
import contest.text_display as text_display


CODE = {'North': 'N', 'South': 'S', 'East': 'E', 'West': 'W', 'Stop': 'X'}
ACTION = {v: k for k, v in CODE.items()}


def name_of(path):
    return os.path.basename(os.path.dirname(path)) if path.endswith('my_team.py') else os.path.basename(path)


def build_layout(name):
    if name.startswith('RANDOM'):
        lname, text = capture.random_layout(seed=int(name[6:]))
        return layout_mod.Layout(layout_name=lname, layout_text=text.split('\n'))
    return layout_mod.get_layout(os.path.join(os.path.dirname(capture.__file__), 'layouts', name))


def side(width, x):
    return 'red' if x < width // 2 else 'blue'


def analyse(game, lay):
    """Replay move history from the initial state and collect events."""
    rules = capture.CaptureRules()
    rules.quiet = True
    state = rules.new_game(lay, [None] * 4, text_display.NullGraphics(), 1200, False, False).state
    width = state.get_walls().width
    starts = [state.get_initial_agent_position(i) for i in range(4)]
    stats = [dict(deaths=0, lost=0, kills=0, returned=0, eaten=0, pac_turns=0, home_turns=0,
                  capsules=0, scared_kills=0, max_carry=0) for _ in range(4)]
    timeline = []
    events = []
    prev = [state.get_agent_state(i).copy() for i in range(4)]
    caps = set(state.get_capsules())
    for step, (idx, action) in enumerate(game.move_history):
        state = state.generate_successor(idx, action)
        cur = [state.get_agent_state(i) for i in range(4)]
        new_caps = set(state.get_capsules())
        if len(new_caps) < len(caps):
            stats[idx]['capsules'] += 1
            events.append((step, 'capsule', idx))
        caps = new_caps
        for i in range(4):
            p0, p1 = prev[i].get_position(), cur[i].get_position()
            if p1 == starts[i] and p0 != starts[i] and abs(p0[0] - p1[0]) + abs(p0[1] - p1[1]) > 1:
                stats[i]['deaths'] += 1
                stats[i]['lost'] += prev[i].num_carrying
                killer = idx if i != idx else None
                if i == idx:
                    # Moved into a ghost: the killer is the nearest opponent.
                    opp = [j for j in range(4) if j % 2 != i % 2]
                    killer = min(opp, key=lambda j: abs(cur[j].get_position()[0] - p0[0]) + abs(cur[j].get_position()[1] - p0[1]))
                stats[killer]['kills'] += 1
                if prev[i].is_pacman is False:
                    stats[killer]['scared_kills'] += 1
                events.append((step, 'death', i, prev[i].num_carrying, side(width, p0[0])))
            stats[i]['returned'] += max(0, cur[i].num_returned - prev[i].num_returned)
            if cur[i].num_carrying > prev[i].num_carrying and i == idx:
                stats[i]['eaten'] += cur[i].num_carrying - prev[i].num_carrying
            stats[i]['max_carry'] = max(stats[i]['max_carry'], cur[i].num_carrying)
        stats[idx]['pac_turns' if cur[idx].is_pacman else 'home_turns'] += 1
        if step % 100 == 99:
            timeline.append(state.data.score)
        prev = [s.copy() for s in cur]
    return stats, timeline, events


def play(red, blue, layout_name, seed):
    lay = build_layout(layout_name)
    random.seed(seed)
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        red_agents = capture.load_agents(True, red, {})
        blue_agents = capture.load_agents(False, blue, {})
        agents = [red_agents[0], blue_agents[0], red_agents[1], blue_agents[1]]
        rules = capture.CaptureRules()
        rules.quiet = True
        started = time.time()
        game = rules.new_game(lay, agents, text_display.NullGraphics(), 1200, True, True)
        game.run()
        elapsed = time.time() - started
    stats, timeline, events = analyse(game, lay)
    log = sink.getvalue()
    return dict(red=red, blue=blue, layout=layout_name, seed=seed, score=game.state.data.score,
                length=len(game.move_history), crashed=game.agent_crashed, timeout=game.agent_timeout,
                warnings=log.count('took too long'), seconds=round(elapsed, 1),
                agent_time=[round(t, 2) for t in game.total_agent_times],
                stats=stats, timeline=timeline, events=events,
                moves=''.join(f'{i}{CODE[a]}' for i, a in game.move_history))


if __name__ == '__main__':
    print(json.dumps(play(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))))
