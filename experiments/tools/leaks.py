"""Where does the opponent score against us? Classify every food the opponent banks by our guard situation.

    venv/bin/python -B experiments/tools/leaks.py OUT.jsonl [--team aljaz2]
For each opponent raid that ends at home with food, every food cell eaten in that raid is tagged at the moment
it was eaten: no guard home / guard scared / guard present, with the guard's distance to the food and the
food's depth (steps from the raider's own side).
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from match import ACTION, build_layout, name_of  # noqa: E402
from episodes import bfs_all  # noqa: E402
from contest import capture  # noqa: E402
import contest.text_display as text_display  # noqa: E402


def pos(a):
    p = a.get_position()
    return None if p is None else (int(p[0]), int(p[1]))


def leaks(row, ours_red):
    lay = build_layout(row['layout'])
    rules = capture.CaptureRules()
    rules.quiet = True
    state = rules.new_game(lay, [None] * 4, text_display.NullGraphics(), 1200, False, False).state
    walls = state.get_walls()
    dist = bfs_all(walls)
    mid = walls.width // 2
    ours = [0, 2] if ours_red else [1, 3]
    theirs = [1, 3] if ours_red else [0, 2]
    their_side = [c for c in dist if (c[0] >= mid) == ours_red]
    depth = {c: min(dist[c].get(h, 99) for h in their_side) for c in dist}
    raids = {}
    banked = {}
    out = []
    moves = row['moves']
    for step in range(len(moves) // 2):
        idx, a = int(moves[2 * step]), moves[2 * step + 1]
        food_before = state.get_red_food() if ours_red else state.get_blue_food()
        state = state.generate_successor(idx, ACTION[a])
        if idx not in theirs:
            continue
        me = state.get_agent_state(idx)
        p = pos(me)
        if me.is_pacman:
            r = raids.setdefault(idx, [])
            if food_before[p[0]][p[1]]:
                guards = [state.get_agent_state(k) for k in ours]
                home = [g for g in guards if not g.is_pacman]
                ready = [g for g in home if g.scared_timer == 0]
                if not home:
                    tag = 'no_guard'
                elif not ready:
                    tag = 'guard_scared'
                else:
                    dg = min(dist[p].get(pos(g), 99) for g in ready)
                    tag = 'guard_d<=%d' % (2 if dg <= 2 else 5 if dg <= 5 else 10 if dg <= 10 else 99)
                r.append((tag, depth[p], sum(g.is_pacman for g in guards)))
        elif idx in raids:
            r = raids.pop(idx)
            if me.num_returned > banked.get(idx, 0):
                out.extend(r)
            banked[idx] = me.num_returned
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('rows')
    ap.add_argument('--team', default='aljaz2')
    a = ap.parse_args()
    by_opp = collections.defaultdict(collections.Counter)
    depth_hist = collections.defaultdict(collections.Counter)
    games = collections.Counter()
    for line in open(a.rows):
        row = json.loads(line)
        ours_red = a.team in row['red']
        opp = name_of(row['blue'] if ours_red else row['red'])
        games[opp] += 1
        for tag, dep, our_pacs in leaks(row, ours_red):
            by_opp[opp][tag] += 1
            by_opp[opp]['our_pacs=%d' % our_pacs] += 1
            depth_hist[opp][min(dep, 6)] += 1
    for opp, c in sorted(by_opp.items()):
        n = games[opp]
        total = sum(v for k, v in c.items() if not k.startswith('our_pacs'))
        tags = ' '.join('%s %.1f' % (k, c[k] / n) for k in sorted(c) if not k.startswith('our_pacs'))
        pacs = ' '.join('%s %.1f' % (k, c[k] / n) for k in sorted(c) if k.startswith('our_pacs'))
        deps = ' '.join('d%d:%.1f' % (k, depth_hist[opp][k] / n) for k in sorted(depth_hist[opp]))
        print('%-30s banked/g %.1f | %s | %s | depth %s' % (opp[:30], total / n, tags, pacs, deps))


if __name__ == '__main__':
    main()
