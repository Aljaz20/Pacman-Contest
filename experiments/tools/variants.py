"""Run one team under several PACLAB_PARAMS variants against an opponent pool, then compare.

    venv/bin/python -B experiments/tools/variants.py TEAM.py OUTDIR '{"base": {}, "noslack": {"slack": [0]}}' \
        [--pool dev] [--layouts dev20] [--seeds 1] [--jobs 14]
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REF = 'local-runs/ref'
POOLS = {
    'dev': ['local-runs/ext/l2_team.py', f'{REF}/aljaz_astar_v27_full_state/my_team.py',
            f'{REF}/aljaz_mdp_frontier_v4/my_team.py', f'{REF}/control_double_offense/my_team.py',
            f'{REF}/control_greedy_food/my_team.py', f'{REF}/aljaz_astar_v26_pursuit/my_team.py'],
    'l2': ['local-runs/ext/l2_team.py'],
    'v27': [f'{REF}/aljaz_astar_v27_full_state/my_team.py'],
    # Acceptance pool: dev pool plus our own frozen earlier version (self-play ladder rung).
    'accept': ['local-runs/ext/l2_team.py', f'{REF}/aljaz_astar_v27_full_state/my_team.py',
               f'{REF}/aljaz_mdp_frontier_v4/my_team.py', f'{REF}/control_double_offense/my_team.py',
               f'{REF}/aljaz_astar_v26_pursuit/my_team.py', f'{REF}/aljaz2_v1g/my_team.py'],
    # Wide pool: simple archetypes (course baseline, border campers, greedy eater), hard search teams
    # (L2 and Codex's strongest time-bounded ones) and the Codex teams used so far.
    'wide': [f'{REF}/baseline_team.py', f'{REF}/control_double_defense/my_team.py',
             f'{REF}/control_greedy_food/my_team.py', 'local-runs/ext/l2_team.py',
             f'{REF}/aljaz_expectimax_v37_team_plan_b200/my_team.py', f'{REF}/aljaz_hybrid_v42_turn_guard/my_team.py',
             f'{REF}/aljaz_tactical_uct_v7_b200/my_team.py', f'{REF}/aljaz_astar_v27_full_state/my_team.py',
             f'{REF}/aljaz_astar_v26_pursuit/my_team.py', f'{REF}/aljaz_mdp_frontier_v4/my_team.py'],
    # Codex teams never used for tuning: a check that changes generalise beyond the wide pool.
    'unseen': [f'{REF}/aljaz_minimax_v3/my_team.py', f'{REF}/aljaz_neural_v28_pursuit/my_team.py',
               f'{REF}/aljaz_route_uct_v12_turn_guard_b200/my_team.py'],
}
POOLS['report'] = POOLS['wide'] + POOLS['unseen']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('team')
    ap.add_argument('outdir')
    ap.add_argument('variants')
    ap.add_argument('--pool', default='dev')
    ap.add_argument('--opps', nargs='*', default=None)
    ap.add_argument('--layouts', nargs='+', default=['dev20'])
    ap.add_argument('--seeds', nargs='+', default=['1'])
    ap.add_argument('--jobs', default='14')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    variants = json.loads(a.variants)
    opps = a.opps or POOLS[a.pool]
    files = []
    for name, params in variants.items():
        out = os.path.join(a.outdir, f'{name}.jsonl')
        files.append(out)
        if os.path.exists(out):
            continue
        cmd = [sys.executable, '-B', os.path.join(HERE, 'batch.py'), out, '--teams', a.team, '--opps'] + opps + \
              ['--layouts'] + a.layouts + ['--seeds'] + a.seeds + ['--jobs', a.jobs, '--params', json.dumps(params)]
        subprocess.run(cmd, stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, '-B', os.path.join(HERE, 'compare.py')] + files)


if __name__ == '__main__':
    main()
