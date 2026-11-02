# Working on this repo

A short guide for the two of us. `README.md` describes the project; this file is how we work.

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
- The agents use only the standard library and the course engine. If a package is ever needed, add it to a root
  `requirements.txt`, pinned to a version that installs on Python 3.8 on Linux (the tournament runs Ubuntu).
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

## Teams and experiments

```text
my_team.py                      submission
teams/<team>/my_team.py         complete two-agent team (paclab = the submission; others are test opponents)
experiments/tools/              match runner, comparisons, tournament and report tools
experiments/LOG.md              one row per experiment: change, games, result, decision
experiments/report.html         every development step and per-opponent tables of each version
experiments/tournament.html     final round robin
local-runs/                     results, logs and frozen reference copies (gitignored, not shared)
```

Run the tools from the repository root with `venv/bin/python -B`; each tool's docstring shows its usage.
- `match.py RED.py BLUE.py LAYOUT SEED` plays one game under tournament limits and prints one JSON line
  with the result and per-agent events.
- `variants.py TEAM.py OUTDIR '{"base": {}, "x": {"param": [1]}}' --pool dev --layouts dev20 --jobs 10` plays
  one team under several `PACLAB_PARAMS` settings against an opponent pool. The pools point to frozen copies
  in `local-runs/ref/` and `local-runs/ext/`, kept local so that logged results stay reproducible; most of these
  teams are in `teams/`, a few earlier versions exist only there.
- `paired.py NEW.jsonl OLD.jsonl [OLD2.jsonl ...]` compares two runs game by game (difference in points
  per game and its standard error).
- `tournament.py OUT.jsonl TEAMS.txt --layouts ... --jobs 10` runs a resumable round robin;
  `tournament_report.py` builds its page, `report.py` builds `experiments/report.html`.
- `replay_check.py` replays recorded games and checks that a team still picks every recorded move
  (use it after a cleanup that should not change behaviour).
- `summarize.py`, `episodes.py`, `show.py`, `compare.py`, `leaks.py`, `paired_split.py`, `batch.py` and
  `rerun_stalled.py` summarise runs, split games into raids, print a game step by step and re-run stalled games.

How we decide on a change: compare the new run with a fresh run of the current version at the same machine
load (time-bounded opponents are not deterministic), over both the development maps and unseen maps. Accept it
only if the paired difference in points per game is positive by at least 1.5 standard errors, the unseen maps
are not worse and the simple teams are still beaten. Log the result in `experiments/LOG.md` either way.

To add a team, copy `teams/template/` to `teams/<name>/` and implement `create_team(...)` in `my_team.py`
(keep helper modules inside the folder, with names no other team uses: all teams run in one process).
