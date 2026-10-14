# Working on this repo

A short guide for the two of us. `README.md` is the original template text; this file is how we work.

## Setup (once)
Follow `README.md`. Notes:
- Clone with HTTPS if you have no SSH key: `git clone https://github.com/Aljaz20/Pacman-Contest.git`
- Use Python 3.8 for the venv (the version the course expects).
- If the game window fails with `Can't find a usable init.tcl` (Python installed with `uv`), add to the
  end of `venv/bin/activate`, with your Python's install folder:
  `export TCL_LIBRARY="<python>/lib/tcl8.6"` and `export TK_LIBRARY="<python>/lib/tk8.6"`

## What to edit
- **`my_team.py` in the repo root is the official submission.** It stays there and keeps `create_team(...)`.
  Develop complete experimental teams under `teams/<team-id>/`; adding a team or running a league does
  not replace the official submission. Promotion is a separate reviewed PR.
- Do not edit anything in `pacman-contest/`; that is the course's engine (a git submodule).
- Need a package (e.g. numpy)? Add it to a `requirements.txt` in the repo root, pinned to a version that
  installs on Python 3.8 on Linux (the tournament runs Ubuntu), and install it in your venv.
- Prefer a standalone `my_team.py` inside each experimental team. A helper module needs the `sys.path` import from
  `pacman-contest/src/contest/CONTEST.md` and a name no other team would use (all teams run in one process).

## Running games
```bash
source venv/bin/activate
cd pacman-contest/src/contest
python capture.py -r baseline_team -b ../../../my_team.py           # with graphics
python capture.py -r baseline_team -b ../../../my_team.py -q -c     # no graphics, tournament time limits
python capture.py -r baseline_team -b ../../../my_team.py -l RANDOM7  # random maze, seed 7
python capture.py -r ../../../my_team.py -b baseline_team -l officeCapture  # play as red on another map
python capture.py -r baseline_team -b ../../../my_team.py --record --record-log
python capture.py --replay=www/contest_default/replays/match_0.replay
```
Layouts are in `pacman-contest/src/contest/layouts/`. `--help` lists every option.

## Before opening a PR
- Play on several layouts **and both colours** (the starting colour is randomly chosen) with `-c`. Without `-c` no time limit
  is enforced at all, so a slow agent looks fine locally and forfeits in the tournament.
- No time warnings. Limits: 1 s per move (3 warnings = forfeit), 3 s for any single move = forfeit,
  15 s for `register_initial_state`. Leave headroom; the tournament machine may be slower than ours.
- No `print` in the agent: everything an agent writes to stdout/stderr is published with the results.
- No threads or computing during the opponent's turn (disqualification).
- Compare against `main`, not only `baseline_team`: the question is whether the change is better than
  what we already have.

## Branches and pull requests
- `main` is our submission: the tournament plays whatever is on `main`, so it must always work.
  Nobody pushes to `main` directly.
- Each of us works on our own branch: `aljaz`, `David`. Update it from `main` before starting something new
  (`git pull origin main`).
- A change reaches `main` only through a pull request into `main`. The other one reviews and merges.
- In the PR description: what changed, why, and the results (layouts, colours, wins/ties/losses against
  `main` and `baseline_team`, any time warnings).
- Commit messages: `Type(component): brief description`, with a second message paragraph only if needed.
  Example: `git commit -m "Add(agent): track opponent positions"`.

## Dates
- Test tournament: 23 November. Final run: 4 December. Deadline: Thursday 3 December, 23:59.

## Shared experimental architecture

```text
my_team.py                      official submission, unchanged by experimental runs
teams/<team-id>/my_team.py       complete two-agent candidate
teams/<team-id>/team.json        discovery metadata and factory options
evaluation/core.py              metadata, validation, paired schedules and outcome classification
evaluation/run_tournament.py    source snapshots, parallel processes, checkpoint and resume
evaluation/build_report.py      generated Markdown and standalone HTML reports
evaluation/configs/*.json       explicit tournament rosters and reproducible map/seed suites
reports/<run-id>/                shareable manifest, match records and generated reports
local-runs/<run-id>/             gitignored source snapshots, lock, detailed logs and replays
tests/                          unit tests and isolated real-engine integration test
```

The shared tools require Python 3.8 and the course environment from Setup. No new dependencies are needed.
Run the commands below from the repository root using `venv/bin/python`. Tools are invoked with `-m`;
do not run the module files directly. These tools work on macOS and Linux (POSIX process groups).

## Adding a complete experimental team

1. Copy `teams/template/` to a unique directory, for example `teams/david_mcts/` or `teams/aljaz_astar/`.
2. Edit `team.json`: choose a unique ID, fill in author, description and approach, describe components,
   and set `enabled` to `true`. All built-in IDs shown by discovery are reserved, including `submission`, `baseline` and `course_*`. The disabled template
   wraps the course baseline; it is runnable scaffolding, not a new strategy.
3. Implement `my_team.py`. Expose `create_team(first_index, second_index, is_red, ...)`, returning exactly
   two compatible agents. Defaults must run without extra options. Keep all helper modules and required
   assets inside the team folder; load assets relative to `__file__`. The runner snapshots the whole folder.
4. Keep Python 3.8 compatibility, time limits, academic attribution and the no-thread/no-output rules.
   Agent computation stays within its own turn. Use uniquely named helper modules: both opponents run
   in the same game process. Namespace imports so one team cannot import another team's helper by accident.
   The runner rejects conflicting direct helper module/package names in the selected teams.
5. Run discovery, include the ID in a config, validate with `--dry-run`, and test on both colours.

Example metadata (fields below are a schema example, not an implemented MCTS team):

```json
{
  "id": "david_mcts",
  "author": "David",
  "description": "Describe the actual strategy and its assumptions",
  "approach": "MCTS",
  "components": ["map analysis", "opponent beliefs"],
  "entry": "my_team.py",
  "options": {},
  "enabled": true
}
```

`id` uses lowercase letters, digits, underscore/hyphen, starts with a letter and is at most 64 characters.
`author`, `description`, `approach`, `entry` are nonempty strings. `entry` stays inside the team folder.
`options` is a dictionary of scalar factory keyword arguments; comma, equals and newline in values are
not supported by the engine's option parser. `components` is an optional list of descriptive strings.
Symlinks are rejected in frozen snapshots. Dependencies must follow the root `requirements.txt` policy.
The built-in `submission` snapshots root Python files and expects its assets to be standalone; use a team
folder for asset-backed experiments. `baseline` uses the provided course baseline.

## Configuring and running a tournament

```bash
venv/bin/python -m evaluation.run_tournament --list-teams
venv/bin/python -m evaluation.run_tournament --config evaluation/configs/smoke.json --dry-run
venv/bin/python -m evaluation.run_tournament --config evaluation/configs/smoke.json --jobs 2
venv/bin/python -m evaluation.run_tournament --config evaluation/configs/league.json --jobs 2 --run-id comparison-v1
```

The config explicitly lists participating IDs in `teams`, fixed layouts and/or `RANDOM<positive seed>`
in `layouts`, number of `repeats`, base game `seed`, and `moves` (normally 1200). `schema_version` is 1.
`record` enables local replays; it defaults to false. `game_timeout` defaults to 1800 seconds per game
and prevents a hung subprocess. It is not a limit on the whole tournament. Unknown fields are rejected.
The supplied league roster initially contains only submission and baseline; add your implemented team IDs.

The runner schedules every unordered pair on every layout and repetition, with both colour assignments
using the same game seed. Each layout/repetition uses a common seed across opponents. `RANDOM7` fixes
map generation independently of the game seed. The frozen worker reapplies the game seed after layout
creation, because the course random-map generator otherwise overwrites global RNG state. New manifests
record `seed_policy: post-layout-reset`; older runs retain their original frozen launcher semantics. The match count is:
`teams * (teams - 1) / 2 * layouts * 2 * repeats`.
Layouts and repeats form schedule blocks; partial runs still need coverage checks before ranking.

`--jobs` controls independent game processes and defaults to 1. Calibrate throughput on your machine;
more workers can increase timing warnings under load. Every game enforces the course limits with `-c`.
For reliable finalist timing checks, run with `--jobs 1`. This parallelism is outside the agents and
never permits an agent to use threads or compute during an opponent's turn.

## Soft stop and resume

Press `Ctrl+C` in the runner terminal. The scheduler stops starting new games, finishes active games,
saves their results, generates both reports and exits. Child game processes are in separate process
groups, so the terminal interrupt does not kill the active games. `SIGTERM` also requests a soft stop.
There is no automatic whole-run time limit. Repeated interrupts keep the same soft-stop behaviour.

```bash
venv/bin/python -m evaluation.run_tournament --resume comparison-v1 --jobs 2
venv/bin/python -m evaluation.build_report reports/comparison-v1
```

Resume validates the frozen snapshots and reuses the original roster/config, even if live team files
have since changed. It skips all recorded matches, including recorded errors; investigate an error and
start a new run to test a fix. Unrecorded matches remain pending. A run lock rejects simultaneous runners
for the same ID. After an abrupt OS kill, confirm the recorded PID is no longer running before manually
removing `local-runs/<run-id>/runner.lock`. Do not remove a live runner's lock. If snapshots were deleted,
the shared report remains readable but that run cannot be resumed.

## Reading and sharing results

Each run has immutable candidate source/asset hashes and an engine hash. A manifest records schema version,
run ID, configuration, full schedule, team metadata, repository/engine revisions, dirty status, evaluator
hash, Python/dependency/platform details, and execution sessions with worker counts and timestamps.
Commit SHA alone does not describe uncommitted code; compare hashes as well.

`matches.jsonl` contains one JSON object per recorded match: schedule identifiers, red/blue IDs, layout,
seed, repeat, `status`, signed red `score`, winner colour (or null), return code, elapsed seconds, per-side
warning/timeout/crash counts, session ID, worker count, finish timestamp, diagnostic and local log reference.
Statuses are `completed`, `forfeit` and `error`. Pending matches have no result record. Forfeits retain the
engine's winner and count in W/T/L; runner/load/engine errors with no valid outcome do not count as ties.
Warnings/timeouts are engine-event counts, not measurements of maximum per-action time. Forfeit score
margins come from the engine's synthetic crash result; read reliability alongside average score.

`report.md` is readable in GitHub and PRs. `report.html` works offline in a browser with team/layout/status
filters on individual matches. Both show ranking, pairwise W/T/L, colour/layout breakdowns, score summaries,
uncertainty, coverage and failure counts. HTML filters change the match explorer, not aggregate tables.
Approximate win-proportion intervals assume independent games and treat ties as non-wins; correlated maps
and repeated trials limit that interpretation. Shortened games are labelled validation-only.

Commit a candidate with its comparison configuration and relevant `reports/<run-id>/` folder. Use a unique
run folder for each execution, so the other teammate can add reports without editing yours. Include in the
PR the actual change, rationale, run reference, baseline/current-submission comparison and timing evidence.
Do not combine rankings across different rosters, source versions, suites or load conditions without
checking comparability. Keep detailed logs, snapshots and bulky replays in gitignored `local-runs/`;
share selected diagnostics separately when needed. Adding a candidate does not automatically promote it.

## Verifying the tooling

```bash
venv/bin/python -B -m unittest discover -s tests -v
```

The integration test uses a temporary checkout, six shortened real games, two concurrent workers, a
terminal-group soft stop, snapshot-preserving resume and both report formats. It is infrastructure
validation, not evidence that any strategy is stronger.

## Reference and control opponents

The runner includes `baseline`, plus `course_team_name_1`, `course_team_name_2` and
`course_team_template` from the engine's provided examples. Their executable strategy logic matches
the supplied baseline; they are reference aliases, not three independent algorithm families.
The `control_*` folders supply random legal actions, nearest-food greed, two offensive reflex agents
and two defensive reflex agents. They provide different diagnostic behaviour; weakness is not assumed
for every map or matchup. Do not treat repeated baseline-equivalent references as independent evidence.

Add discovered reference/control IDs and your implemented candidates to a copy of
`evaluation/configs/league.json`. Use `--list-teams` to see available IDs and `--dry-run` to inspect
the configured scope before launching a comparison. No experimental strategy roster is bundled here.
Reliability checks against baseline are separate from selecting a tournament winner:

```bash
venv/bin/python -m evaluation.validate_candidates submission control_greedy_food --prefix check-new --jobs 2
venv/bin/python -m evaluation.validate_candidates submission --prefix timing-new --jobs 1 --layouts jumboCapture
```

The validation command returns failure for missing matches, crashes, forfeits or timing events. A prefix
identifies frozen candidate versions; after changing code/assets/config, choose a new prefix rather than
resuming an old validation and attributing its results to current code.
