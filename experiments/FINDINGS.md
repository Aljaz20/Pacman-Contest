# aljaz2 - investigation findings (2026-10-28)

Measured with `experiments` tools (own harness, full game replay, per-raid episode analysis).
Opponent L2 = private file from RF34684 (kept out of git in `local-runs/ext/`).
Pool: 10 fixed layouts + RANDOM1-10, both colours.

## Headline numbers
| team | games vs L2 | win | margin | our kills/g | L2 deaths/g | L2 food lost at death/g |
|---|---|---|---|---|---|---|
| A-Star v27 full_state | 80 (2 seeds) | ~50% | ~0 | 0.5 | 0.5 | 9.2 |
| Expectimax v37 b200 | 26 (partial) | 38% | -0.3 | 1.2 | 1.2 | 15.9 |

Both teams return ~21-23 food per game. Games are pure food races: almost nobody gets caught.

## How L2 actually scores
- L2's attacker never goes home voluntarily (only when it sees a ghost, when <=2 food remain or time is short).
  It eats until chased, so it carries huge loads: 37 raids with >=8 food in 40 games, often 15-40,
  twice 115+ food (officeCapture) in ONE raid.
- During those big raids our defender was usually right next to it: min distance 1-2,
  "within 3 squares" for 100-900 consecutive steps - and still never caught it.
- On RANDOM maps the other failure dominates: no defender at home at all (both our agents attacking,
  defender-present 0-30% of the raid).

## Structural causes (code + replays)
1. Defence = chase the invader's current cell (A* to `defender_assignments` target). Equal speed in a maze
   with loops => the chaser trails forever. No interception (cut the escape route / border exit),
   no pincer with the teammate, no "invader carries 30 food -> both hunt" rule.
2. No dedicated defender. Roles come from one shared `choose_goal`: both agents collect by default;
   defence only triggers for an agent that happens to be a ghost at home.
3. Standoff deadlock: replay officeCapture - our pacman with 9 food stands next to L2's ghost for ~750
   steps (both frozen). No progress detection / task abort, one agent wasted for most of the game.
4. Danger model `risk_at` = linear decay inside Manhattan radius 5. No escape-race / dead-end trap reasoning
   (a ghost 6 squares away from a 10-deep pocket counts as zero risk; a ghost behind us counts as risk).
5. All ~200 variants (A*, expectimax, minimax, MCTS, MDP, Q, neural, evolution...) sit on the SAME
   `PacLabAgent.choose_goal` + `risk_at` core. Changing the search family only changes how we walk to a goal
   chosen by the same weak strategy layer -> no family can fix 1-3. Explains why v27..v47 never improved.
6. Evaluation noise: decisions taken from 8-12 games per pair vs L2 (e.g. "5/0/7 vs 7/0/5").
   95% CI at n=12 is +-28 percentage points.
7. Full-state access is already used (observation_function override) and still loses to an agent that
   only sees 5 squares -> information is not the bottleneck, decisions are.

## Last year's winner (RF34684 / article)
Classical, three layers: graph + path finding; behaviour trees per situation (attack/defence);
top strategy layer assigning roles (attack / defend / split). Key: thoroughness and hundreds of games per change.
He also says: most effort went into good rules; full state did not matter much for him.

## Codex history digest (from .claude/notes + reports, recounted by a read-only subagent)
- Root `my_team.py` (the real submission on main) is still the course baseline. Nothing was ever promoted.
- ~V1-V80 in two weeks: 13 families, a 52-team league (~40k games), many 600-1,860-game campaigns.
  Every late campaign (V68 mechanisms, V78 three-level controller, V79 nine variants, V80 behaviour theories)
  ended with "no broad promotion".
- A-Star v27 vs L2: fixed maps ~62%, RANDOM maps ~32% (Codex data, recount); my run: fixed 9W/1D/10L, random 7W/13L.
  RF34684 measured 34.9% over 800 games (likely random maps).
- Codex's own loss diagnosis: 10/11 A* and 13/13 Expectimax losses had ZERO own food lost -> "enemy banks more",
  yet no dedicated-defender redesign followed; the sentry test was 4 maps / tiny n.
- Expectimax hit its planning deadline in 4,527 of 9,050 instrumented decisions (search rarely completes).
- 23 of 28 own deaths were empty Pacmen walking into a ghost (reply guards skip cargo == 0).
- Most V79/V80 L2 verdicts used 8-12 games per pair -> noise.
- Full-state reading of opponents is not checked against course rules (PDF 2.5 says 5-square visibility).
