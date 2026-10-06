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
- **`my_team.py` in the repo root.** The tournament loads exactly this file, so it must stay there and
  keep `create_team(...)`.
- Do not edit anything in `pacman-contest/`; that is the course's engine (a git submodule).
- Need a package (e.g. numpy)? Add it to a `requirements.txt` in the repo root, pinned to a version that
  installs on Python 3.8 on Linux (the tournament runs Ubuntu), and install it in your venv.
- Prefer keeping the agent in `my_team.py`. A helper module needs the `sys.path` import from
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
- Play on several layouts **and both colours** (red moves first) with `-c`. Without `-c` no time limit
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
- Commit messages: `[add] ...`, `[fix] ...`, `[update] ...`, e.g. `[add] Particle filter for enemy tracking`.

## Dates
- Test tournament: 23 November. Final run: 4 December. Deadline: Thursday 3 December, 23:59.
