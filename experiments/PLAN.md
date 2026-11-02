# aljaz2 - brainstorm and plan

Decisions (Aljaz, 2026-10-28): use full state now; new agent from scratch; Codex stopped;
local commits on aljaz2 and multi-hour runs allowed; no push. No compiled packages: only plain Python 3.8 source, tested with tournament limits (-q -c).

## Goal (measurable, checked with experiments tools)
- G1 >= 75% wins vs L2 over >= 400 games (10 fixed + 30 RANDOM maps, both colours).
- G2 >= 70% wins head-to-head vs Codex A-Star v27 (current best).
- G3 0 crashes / 0 timeouts; slowest move < 0.25 s locally (cluster safety); start-up < 5 s on jumbo.
- G4 >= 95% vs baseline_team.
- Then: compiled Linux/py3.12 build for RF34684's pool (my_team, Pac*, Tian, ...) - the real generality test.
- Overfitting guard: development maps RANDOM 1-40, holdout RANDOM 5001-5040 used only for acceptance.

## Options considered
| option | verdict |
|---|---|
| Keep patching v27/v37 (2,300-line mixin stack, same choose_goal/risk core) | rejected: 80 versions show the core caps it |
| Another search family (MCTS/expectimax) on top | rejected: search was never the bottleneck; v37 misses deadlines half the time |
| Learning (Q/neural/evolution/self-play RL) | rejected for now: tried, transfer gap, ~5 weeks left |
| **New clean agent, 3 layers (like last year's winner)** | **chosen** - small, explainable, fast, testable |

## Architecture (single file `teams/aljaz2_*/my_team.py`, later root `my_team.py`)
1. **Map layer** (register_initial_state, < 5 s): all-pairs maze distances; border/entry cells; dead-end
   pockets with their exit cell and depth (articulation-point analysis); choke points between enemy border
   and our food; food clusters.
2. **Perception layer**: one interface `where(enemy)` -> exact cell or belief. Two implementations:
   true state (if allowed) and belief tracking (visible 5 squares + noisy distance + food-disappearance).
3. **Behaviour layer** (deterministic rules + small searches):
   - Attack: food chosen by an *escape race*: take food f only if I can reach f and then a safe exit
     (border cell or capsule) strictly before every non-scared ghost can intercept; pockets need
     depth-aware margins. Return when carrying * danger > value of next food; always return when the
     game can be decided. Capsules as escape and as a timed "free raid". Eat scared ghosts if cheap.
   - Defend: *intercept, do not chase*: aim at the invader's escape route (the border cell it must reach,
     or the pocket exit), shadow the enemy attacker along our border before it enters, keep distance 2 when
     scared. Pincer with the teammate when the invader carries a lot.
   - Close combat (distance <= 6 to an enemy): tiny alpha-beta over exact positions, depth 4-6, < 30 ms.
   - Anti-stall: progress watchdog; a standoff longer than N steps triggers a different entry / role switch.
4. **Strategy layer** (team controller, every turn): roles attacker / defender / both attack / both defend
   from score, time left, enemy attackers count and their cargo, our cargo, food left. Example rules:
   1A+1D default; invader with cargo >= 5 -> nearest agent hunts, partner cuts exit; leading late -> 2D;
   enemy has no attacker for long -> 2A split by map halves; behind late -> 2A.

## Evaluation (own fast harness, 15 cores)
- Pool: L2, baseline, A-Star v27, Expectimax v37 (few games, slow), own previous versions, plus probe bots
  (double attacker, double defender/camper, greedy) to test generality, not only L2.
- Per change: paired games on identical maps/seeds/colours vs the previous version; accept only if the
  win-rate gain is outside noise (>= 200 games per comparison, SPRT-style stop). Mechanism metrics
  (kills, deaths, food lost, defender presence, stall time) explain *why*, replays (show.py) confirm it.
- Every accepted version is frozen as a ladder rung; experiments logged in experiments/LOG.md.

## Order of work
1. Map + perception + simple roles skeleton; beats baseline, no timeouts.
2. Defence first (biggest measured leak: L2 banks 15-115 per raid next to our defender).
3. Safe attack with escape race + return policy.
4. Strategy controller (role switching), anti-stall.
5. Close-combat search, capsules/scared logic.
6. Tune on dev maps, confirm on holdout, timing on slow CPU (throttled run).
