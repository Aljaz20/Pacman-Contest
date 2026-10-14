"""Run a resumable league in isolated game processes with course time limits."""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from evaluation.core import ID, classify, digest, discover_teams, read_json, schedule, validate_config, write_json
from evaluation.build_report import build_report, build_stats, load_rows

ROOT = Path(__file__).resolve().parent.parent
SKIP = {'__pycache__', '.git', 'venv', '.venv', 'www', '.DS_Store'}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def copy_tree(source, target):
    target.mkdir(parents=True, exist_ok=True)
    for p in sorted(source.iterdir()):
        if p.name in SKIP or p.suffix in ('.pyc', '.pyo'):
            continue
        if p.is_symlink():
            raise ValueError('Snapshots do not accept symlinks: ' + str(p))
        if p.is_dir():
            copy_tree(p, target / p.name)
        elif p.is_file():
            shutil.copy2(str(p), str(target / p.name))


def tree_hash(folder):
    contents = {}
    for p in sorted(folder.rglob('*')):
        if p.is_symlink():
            raise ValueError('Snapshot contains a symlink: ' + str(p))
        if p.is_file():
            contents[p.relative_to(folder).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return digest(contents)



def check_helpers(folder, teams):
    names = {}
    for tid, t in teams.items():
        entry = folder / tid / t['entry']
        visible = {p.stem for p in entry.parent.glob('*.py') if p != entry and p.name != '__init__.py'}
        visible.update(p.name for p in entry.parent.iterdir() if p.is_dir() and (p / '__init__.py').is_file())
        names[tid] = visible
    for a in names:
        for b in names:
            if a < b and names[a] & names[b]:
                raise ValueError('Conflicting helper imports for {} and {}: {}'.format(a, b, ', '.join(sorted(names[a] & names[b]))))


def verify_snapshots(local, manifest):
    if tree_hash(local / 'engine') != manifest['engine_hash']:
        raise ValueError('Frozen engine snapshot changed or is missing')
    for tid, t in manifest['teams'].items():
        if tree_hash(local / 'teams' / tid) != t['hash']:
            raise ValueError('Frozen team snapshot changed or is missing: ' + tid)


def git(*args):
    p = subprocess.run(['git', '-C', str(ROOT)] + list(args), capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else None


def prepare(run_id, config, workers):
    available = discover_teams(ROOT)
    missing = set(config['teams']) - set(available)
    if missing:
        raise ValueError('Unknown teams: ' + ', '.join(sorted(missing)))
    original = ROOT / 'pacman-contest' / 'src' / 'contest'
    for layout in config['layouts']:
        if not layout.startswith('RANDOM') and not (original / 'layouts' / (layout + '.lay')).is_file():
            raise ValueError('Missing layout: ' + layout)
    report = ROOT / 'reports' / run_id
    local = ROOT / 'local-runs' / run_id
    if report.exists() or local.exists():
        raise ValueError('Run ID already exists; use --resume or choose a new ID')
    local.mkdir(parents=True)
    try:
        frozen = local / 'engine' / 'contest'
        frozen.mkdir(parents=True)
        for p in original.glob('*.py'):
            if p.is_symlink():
                raise ValueError('Engine source symlinks are not supported')
            shutil.copy2(str(p), str(frozen / p.name))
        copy_tree(original / 'layouts', frozen / 'layouts')
        shutil.copy2(str(ROOT / 'evaluation/game_worker.py'), str(local / 'engine/worker.py'))
        teams = {}
        for tid in config['teams']:
            t = dict(available[tid])
            target = local / 'teams' / tid
            if tid in ('submission', 'baseline'):
                target.mkdir(parents=True)
                source = ROOT / t['folder']
                files = [source / t['entry']] if tid == 'baseline' else sorted(source.glob('*.py'))
                for p in files:
                    if p.is_symlink():
                        raise ValueError('Team source symlinks are not supported')
                    shutil.copy2(str(p), str(target / p.name))
            else:
                copy_tree(ROOT / t['folder'], target)
            t['hash'] = tree_hash(target)
            teams[tid] = t
        check_helpers(local / 'teams', teams)
        try:
            from importlib.metadata import version
            dep = version('frozendict')
        except Exception:
            dep = 'unavailable'
        manifest = dict(schema_version=1, seed_policy='post-layout-reset', run_id=run_id, created_at=now(), config=config,
                        teams=teams, schedule=schedule(config['teams'], config['layouts'], config['repeats'], config['seed']),
                        workers=workers, tool_hash=tree_hash(ROOT / 'evaluation'), engine_hash=tree_hash(local / 'engine'), engine_commit=git('-C', str(ROOT / 'pacman-contest'), 'rev-parse', 'HEAD'),
                        repo_commit=git('rev-parse', 'HEAD'), repo_dirty=bool(git('status', '--porcelain')),
                        python=sys.version, dependency_versions={'frozendict': dep},
                        platform=platform.platform(), cpu_count=os.cpu_count(), sessions=[])
        report.mkdir(parents=True)
        write_json(report / 'manifest.json', manifest)
        build_report(report)
        return manifest
    except BaseException:
        shutil.rmtree(str(local))
        if report.exists():
            shutil.rmtree(str(report))
        raise


def play(job, manifest, local, workers, session_id):
    started = time.monotonic()
    work = local / 'matches' / str(job['match_id'])
    work.mkdir(parents=True, exist_ok=True)
    score_path = work / 'www' / 'contest_game' / 'scores' / ('match_{}.json'.format(job['match_id']))
    if score_path.exists():
        score_path.unlink()  # an interrupted attempt must not supply a stale score
    launcher = local / 'engine' / 'worker.py'
    if not launcher.exists():
        launcher = local / 'engine' / 'contest' / 'capture.py'  # preserve frozen legacy run semantics
    cmd = [sys.executable, '-B', str(launcher),
           '-q', '-c', '--delay-step', '0', '-l',
           job['layout'] if job['layout'].startswith('RANDOM') else str(local / 'engine/contest/layouts' / job['layout']),
           '-i', str(manifest['config']['moves']),
           '--setRandomSeed', str(job['seed']), '-u', 'game', '-m', str(job['match_id']),
           '--red-name', 'Red', '--blue-name', 'Blue']
    for side, flag, opts_flag in [('red', '-r', '--redOpts'), ('blue', '-b', '--blueOpts')]:
        t = manifest['teams'][job[side]]
        cmd.extend([flag, str(local / 'teams' / job[side] / t['entry'])])
        if t['options']:
            cmd.extend([opts_flag, ','.join(k + '=' + str(v) for k, v in sorted(t['options'].items()))])
    if manifest['config']['record']:
        cmd.append('--record')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    env['PYTHONPATH'] = str(local / 'engine')  # always import the frozen course package
    log_path = work / 'game.log'
    error = None
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(cmd, cwd=str(work), env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=manifest['config']['game_timeout'])
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            code = process.wait()
            error = 'Game exceeded subprocess timeout'
    text = log_path.read_text(encoding='utf-8', errors='replace')
    score = None
    if score_path.exists():
        try:
            data = read_json(score_path)['games'][0]
            if data[4] not in ('Red', 'Blue', None) or type(data[3]) not in (int, float):
                raise ValueError('Unexpected engine score')
            score = data[3] if data[4] == 'Red' else -data[3] if data[4] == 'Blue' else 0
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            error = 'Invalid score: ' + str(exc)
    outcome = classify(score, text, code)
    if outcome['status'] == 'error' and not error:
        error = 'Missing score or nonzero exit; inspect local game.log'
    return dict(job, **outcome, schema_version=1, elapsed=round(time.monotonic() - started, 3),
                workers=workers, session_id=session_id, finished_at=now(), error=error,
                log='local-runs/{}/matches/{}/game.log'.format(manifest['run_id'], job['match_id']))


def run(manifest, workers):
    run_id = manifest['run_id']
    report, local = ROOT / 'reports' / run_id, ROOT / 'local-runs' / run_id
    lock = local / 'runner.lock'
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise ValueError('Run is locked; check for a live runner before removing local runner.lock')
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(str(os.getpid()))
        verify_snapshots(local, manifest)
        rows = load_rows(report / 'matches.jsonl')
        build_stats(manifest, rows)  # reject duplicates/corrupted schedule before doing any work
        done = {r['match_id'] for r in rows}
        pending = iter(j for j in manifest['schedule'] if j['match_id'] not in done)
        stopped = [False]
        def soft_stop(signum, frame):
            stopped[0] = True
        previous = {sig: signal.signal(sig, soft_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        session = dict(id=len(manifest['sessions']), started_at=now(), workers=workers)
        manifest['sessions'].append(session)
        write_json(report / 'manifest.json', manifest)
        print('Run {}: {} pending matches, {} workers. Ctrl+C finishes active games.'.format(run_id, len(manifest['schedule']) - len(done), workers), flush=True)
        try:
            with (report / 'matches.jsonl').open('a', encoding='utf-8') as output, concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                active = {}
                exhausted = False
                announced = False
                while active or (not exhausted and not stopped[0]):
                    while len(active) < workers and not stopped[0] and not exhausted:
                        job = next(pending, None)
                        if job is None:
                            exhausted = True
                            break
                        active[pool.submit(play, job, manifest, local, workers, session['id'])] = job
                    if stopped[0] and not announced:
                        print('Soft stop requested; waiting for {} active games.'.format(len(active)), flush=True)
                        announced = True
                    if not active:
                        break
                    finished, _ = concurrent.futures.wait(active, timeout=0.2, return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in finished:
                        job = active.pop(future)
                        try:
                            result = future.result()
                        except Exception as exc:
                            result = dict(job, **classify(None, '', -1), schema_version=1,
                                          elapsed=0, workers=workers, session_id=session['id'], finished_at=now(),
                                          error=type(exc).__name__ + ': ' + str(exc), log=None)
                        output.write(json.dumps(result, sort_keys=True) + '\n')
                        output.flush()
                        os.fsync(output.fileno())
                        rows.append(result)
                        print('Match {}: {} vs {} {} score={}'.format(job['match_id'], job['red'], job['blue'], result['status'], result['score']), flush=True)
            session.update(ended_at=now(), soft_stopped=stopped[0])
            write_json(report / 'manifest.json', manifest)
            build_report(report)
            print('Reports: reports/{}/report.md and report.html'.format(run_id), flush=True)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    finally:
        lock.unlink()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    choice = p.add_mutually_exclusive_group(required=True)
    choice.add_argument('--config', type=Path)
    choice.add_argument('--resume', metavar='RUN_ID')
    choice.add_argument('--list-teams', action='store_true')
    p.add_argument('--jobs', type=int, default=1, help='parallel game processes (default: 1)')
    p.add_argument('--run-id', help='unique ID for a new run')
    p.add_argument('--dry-run', action='store_true', help='validate and show schedule size without writing files')
    args = p.parse_args()
    try:
        if args.jobs < 1:
            raise ValueError('--jobs must be >= 1')
        if args.list_teams:
            if args.run_id or args.dry_run:
                raise ValueError('--list-teams cannot use --run-id or --dry-run')
            for tid, t in sorted(discover_teams(ROOT).items()):
                print('{}: {} ({})'.format(tid, t['description'], t['author']))
            return
        if args.resume:
            if args.run_id or args.dry_run or not ID.fullmatch(args.resume):
                raise ValueError('Invalid resume ID or incompatible options')
            manifest = read_json(ROOT / 'reports' / args.resume / 'manifest.json')
            if manifest['run_id'] != args.resume or manifest.get('schema_version') != 1:
                raise ValueError('Manifest ID/version mismatch')
        else:
            c = validate_config(read_json(args.config))
            available = discover_teams(ROOT)
            if set(c['teams']) - set(available):
                raise ValueError('Config references unknown teams')
            original = ROOT / 'pacman-contest/src/contest/layouts'
            for layout in c['layouts']:
                if not layout.startswith('RANDOM') and not (original / (layout + '.lay')).exists():
                    raise ValueError('Missing layout: ' + layout)
            if args.dry_run:
                print(json.dumps(dict(config=c, games=len(schedule(c['teams'], c['layouts'], c['repeats'], c['seed'])), workers=args.jobs), indent=2))
                return
            run_id = args.run_id or 'run-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:8]
            if not ID.fullmatch(run_id):
                raise ValueError('Run ID must use lowercase letters, digits, hyphen/underscore and start with a letter')
            manifest = prepare(run_id, c, args.jobs)
        run(manifest, args.jobs)
    except (ValueError, OSError, KeyError) as exc:
        p.exit(2, 'Error: ' + str(exc) + '\n')


if __name__ == '__main__':
    main()
