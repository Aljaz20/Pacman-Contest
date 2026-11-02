"""Raid/invasion episode analysis over saved move histories (rows with 'moves').

    venv/bin/python -B local-runs/lab/episodes.py OUT.jsonl
For every pacman episode (agent crosses into enemy half until it is home again or dies) we record
food eaten, outcome, and how the defending team was positioned during the episode.
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from match import ACTION, build_layout, name_of  # noqa: E402
from contest import capture  # noqa: E402
import contest.text_display as text_display  # noqa: E402

def bfs_all(walls):
    cells = [(x, y) for x in range(walls.width) for y in range(walls.height) if not walls[x][y]]
    cs = set(cells)
    dist = {}
    for c in cells:
        d = {c: 0}
        q = collections.deque([c])
        while q:
            p = q.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (p[0] + dx, p[1] + dy)
                if n in cs and n not in d:
                    d[n] = d[p] + 1
                    q.append(n)
        dist[c] = d
    return dist


def episodes(row):
    lay = build_layout(row['layout'])
    rules = capture.CaptureRules()
    rules.quiet = True
    state = rules.new_game(lay, [None] * 4, text_display.NullGraphics(), 1200, False, False).state
    dist = bfs_all(state.get_walls())
    eps = []
    open_ep = {}
    prev = [state.get_agent_state(i).copy() for i in range(4)]
    moves = row['moves']
    for step in range(0, len(moves) // 2):
        idx, a = int(moves[2 * step]), moves[2 * step + 1]
        action = ACTION[a]
        state = state.generate_successor(idx, action)
        cur = [state.get_agent_state(i).copy() for i in range(4)]
        for j in range(4):
            if cur[j].is_pacman and j not in open_ep:
                open_ep[j] = dict(agent=j, start=step, eaten=0, defender_home=0, steps=0, min_def=99,
                                  close_steps=0)
            if j in open_ep:
                e = open_ep[j]
                e['steps'] += 1
                e['eaten'] = max(e['eaten'], cur[j].num_carrying)
                pj = tuple(int(v) for v in cur[j].get_position())
                defenders = [k for k in range(4) if k % 2 != j % 2 and not cur[k].is_pacman
                             and cur[k].scared_timer == 0]
                if defenders:
                    e['defender_home'] += 1
                    dmin = min(dist[pj].get(tuple(int(v) for v in cur[k].get_position()), 99) for k in defenders)
                    e['min_def'] = min(e['min_def'], dmin)
                    e['close_steps'] += dmin <= 3
                if not cur[j].is_pacman:
                    got = cur[j].num_returned - prev[j].num_returned
                    p0 = prev[j].get_position()
                    died = p0 is not None and abs(p0[0] - cur[j].get_position()[0]) + abs(p0[1] - cur[j].get_position()[1]) > 1
                    e['end'] = step
                    e['outcome'] = 'died' if died else 'returned'
                    e['returned'] = got
                    e['lost'] = prev[j].num_carrying if died else 0
                    eps.append(e)
                    del open_ep[j]
        prev = cur
    for j, e in open_ep.items():
        e['end'], e['outcome'], e['returned'], e['lost'] = len(moves) // 2, 'timeout', 0, prev[j].num_carrying
        eps.append(e)
    return eps


def main():
    rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    agg = collections.defaultdict(collections.Counter)
    for r in rows:
        if 'moves' not in r:
            continue
        for e in episodes(r):
            raider_colour = 'red' if e['agent'] % 2 == 0 else 'blue'
            raider = name_of(r[raider_colour])
            c = agg[raider]
            c['games_x'] += 0
            c['eps'] += 1
            c[e['outcome']] += 1
            c['ret_food'] += e['returned']
            c['lost_food'] += e['lost']
            c['eaten'] += e['eaten']
            c['steps'] += e['steps']
            c['no_defender_frac'] += 1 - e['defender_home'] / max(1, e['steps'])
            c['close'] += e['close_steps'] > 0
            if e['outcome'] == 'returned' and e['returned'] > 0:
                c['scoring_eps'] += 1
                c['scoring_no_def'] += (1 - e['defender_home'] / max(1, e['steps']))
                c['scoring_min_def'] += min(e['min_def'], 20)
                c['scoring_close'] += e['close_steps'] > 0
        for colour in ('red', 'blue'):
            agg[name_of(r[colour])]['games'] += 1
    print('raider | games | raids/g | returned died timeout | food ret/raid | lost/raid | '
          'avg defender-absent% | scoring raids: n/g, def-absent%, avg min def dist, %ever-close')
    for k, c in sorted(agg.items()):
        g = max(1, c['games'])
        n = max(1, c['eps'])
        s = max(1, c['scoring_eps'])
        print(f"{k[:36]:36} | {c['games']} | {c['eps']/g:.1f} | {c['returned']} {c['died']} {c['timeout']} | "
              f"{c['ret_food']/n:.2f} | {c['lost_food']/n:.2f} | {100*c['no_defender_frac']/n:.0f}% | "
              f"{c['scoring_eps']/g:.1f}, {100*c['scoring_no_def']/s:.0f}%, {c['scoring_min_def']/s:.1f}, "
              f"{100*c['scoring_close']/s:.0f}%")


if __name__ == '__main__':
    main()
