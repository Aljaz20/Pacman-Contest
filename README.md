# PacLab - Pacman Capture the Flag agent

Team PacLab's agent for the Autonomous Systems (UPF) Pacman contest. Team members are in `TEAM.md`;
how we work on the repo is in `CONTRIBUTING.md`.

## The submission
`my_team.py` in the repository root is the submission. The same agent is also in `teams/paclab/my_team.py`,
so it can play against the other teams under one name.

The agent has three layers. The maze layer precomputes all maze distances and the border. The attacker
keeps an escape route: for every cell it knows the earliest move an enemy ghost can stand there and the latest
move it may stand there and still reach home (or a capsule) ahead of every ghost, and it only moves inside
that limit. The defender shrinks the region an invader can reach and shadows the border when nobody invades.
The team layer assigns the roles from the score, the time left, the invaders and the food carried.

## Setting up
1. `git clone https://github.com/Aljaz20/Pacman-Contest.git` and `cd Pacman-Contest`
2. `git submodule update --init --remote` (pulls the course engine into `pacman-contest/`)
3. `python3.8 -m venv venv` and `source venv/bin/activate`
4. `cd pacman-contest && pip install -r requirements.txt && pip install -e . && cd ..`

No packages beyond the course requirements are needed.

## Running a game
```bash
cd pacman-contest/src/contest
python capture.py -r baseline_team -b ../../../my_team.py -q -c                  # tournament time limits
python capture.py -r ../../../teams/l2/my_team.py -b ../../../my_team.py -l RANDOM7  # with graphics
```
`-c` enforces the tournament time limits; always test with it.

## Teams
Every folder in `teams/` is a complete two-agent team that `capture.py` can load. Rank and points per game
(win 3, draw 1) are from our final round robin of 20 teams (see Results).

**`paclab`** (rank 1, 2.29) - our final agent, identical to the root `my_team.py` (described above).

**Our earlier agents** (`astar` ... `hybrid_particle`). Before `paclab` we built a family of agents that share one
core and differ in how they pick a move. For each family the folder holds the version that did best in our tests,
byte for byte as it played in the tournament. The shared core: both agents track each enemy with a belief over
its position (from sight and the noisy distance readings), estimate the risk of a cell from enemy ghosts within
a few steps, and choose a goal each move (collect food, return home when the cargo or the risk is too high,
defend against an invader, guard the border late in a won game, or take a capsule when threatened). The family
decides how to get to that goal:

| folder | rank, pts/game | how it picks a move |
|---|---|---|
| `astar` | 4, 1.89 | A* path to the goal, avoiding risky cells. Reads the full game state (exact enemy positions), and checks for three own moves that a visible pursuer cannot catch it on the way home |
| `qlearning` | 8, 1.60 | Linear Q-function: the move with the best weighted sum of 19 features (food, danger, dead ends, return deadline...); weights learned by Q-learning in training games, in `model.json`. Same pursuit check as `astar` |
| `neural` | 9, 1.57 | Small neural network (18 features, 24 hidden units) scores each move; trained offline to imitate our search planner, in `model.json`. Same pursuit check and the reply check below |
| `mdp_frontier` | 10, 1.55 | Value iteration on a local MDP around the agent; a far goal is replaced by a reachable cell on the edge of that area |
| `expectimax` | 11, 1.55 | Expectimax search with nearby enemies as chance nodes, with a team layer that plans joint tasks and aborts them; reads the full game state |
| `route_uct` | 12, 1.50 | Monte Carlo tree search (UCT) guided by a precomputed route, plus a check that no visible enemy reply can catch an agent that carries food |
| `tactical_uct` | 13, 1.49 | The same UCT search without the route guide, with the same reply check |
| `hybrid` | 14, 1.40 | Switches per move: minimax or expectimax when an enemy is near, belief-based MCTS when risk is high, a goal planner otherwise; same reply check |
| `minimax` | 15, 1.37 | Alpha-beta minimax over nearby enemies (worst-case replies), A* when no enemy is near |
| `hybrid_particle` | 17, 1.24 | The `hybrid` switch with a particle filter for enemy positions instead of exact beliefs |

**`l2`** (rank 7, 1.72) - external reference team (an offensive and a defensive reflex agent with A* paths that avoid enemies) that
we used only as an opponent; not our code.

**Control teams** - simple opponents to check that a change does not lose to easy play:
`control_greedy_food` (rank 18, 0.48: both agents eat the nearest food and go home with 8),
`control_double_defense` (rank 20, 0.07: two defensive baseline agents camp at the border),
`control_double_offense` (two offensive baseline agents), `control_random` (random legal moves) and
`template` (the course baseline, rank 19 with 0.25), a starting point for a new team.

## Results
In our final round robin (20 teams, 34 maps, both colours, 12,920 games, no errors or time warnings) the final
agent ranked 1st with 2.29 points per game (win 3, draw 1). `experiments/tournament.html` has the standings and
head-to-head tables; `experiments/report.html` shows each step of development, what helped and what did not.

## Experiments
`experiments/` holds the tools we used to measure changes (match runner, paired comparisons, tournament,
reports), the log of every experiment (`LOG.md`) and the findings. See `CONTRIBUTING.md` for how to run them.
