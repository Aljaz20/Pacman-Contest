"""Team discovery, deterministic schedules and outcome classification."""
import hashlib
import itertools
import json
import math
import re
from pathlib import Path

ID = re.compile(r'^[a-z][a-z0-9_-]{0,63}$')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read_json(path):
    with Path(path).open(encoding='utf-8') as f:
        return json.load(f)


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    temp.replace(path)


def contained(folder, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError('Expected a relative team file path')
    path = (folder / relative).resolve()
    try:
        path.relative_to(folder.resolve())
    except ValueError:
        raise ValueError('Team entry must stay inside its own folder')
    if not path.is_file():
        raise ValueError('Missing team file: ' + relative)
    return path


def discover_teams(root):
    root = Path(root)
    teams = {
        'submission': dict(id='submission', author='PacLab', description='Current root submission',
                           approach='Current submission', entry='my_team.py', folder='.', options={}),
        'baseline': dict(id='baseline', author='Course staff', description='Provided reflex baseline',
                         approach='Reflex', entry='baseline_team.py',
                         folder='pacman-contest/src/contest', options={}),
    }
    for name in ('team_name_1', 'team_name_2', 'team_template'):
        tid = 'course_' + name
        folder = 'pacman-contest/src/contest/agents/' + name
        if (root / folder / 'my_team.py').exists():
            teams[tid] = dict(id=tid, author='Course staff', description='Provided sample (baseline-equivalent)',
                              approach='Course reflex reference', entry='my_team.py', folder=folder,
                              options={}, category='reference', equivalent_to='baseline')
    for p in sorted((root / 'teams').glob('*/team.json')):
        t = read_json(p)
        if not isinstance(t, dict):
            raise ValueError('Team metadata must be an object: ' + str(p))
        for key in ('id', 'author', 'description', 'approach', 'entry'):
            if not isinstance(t.get(key), str) or not t[key].strip():
                raise ValueError('Missing team field ' + key + ': ' + str(p))
        if not ID.fullmatch(t['id']) or t['id'] in teams:
            raise ValueError('Invalid or duplicate team ID: ' + t['id'])
        contained(p.parent, t['entry'])
        components = t.get('components', [])
        if not isinstance(components, list) or any(not isinstance(v, str) for v in components):
            raise ValueError('components must be a list of strings')
        opts = t.get('options', {})
        if not isinstance(opts, dict) or any(not isinstance(k, str) or not k.isidentifier() or
                not isinstance(v, (str, int, float, bool)) or any(c in str(v) for c in ',=\n')
                for k, v in opts.items()):
            raise ValueError('Options must be scalar values without comma, equals or newline')
        if 'enabled' in t and not isinstance(t['enabled'], bool):
            raise ValueError('enabled must be a boolean')
        if not t.get('enabled', True):
            continue
        t = dict(t, folder=p.parent.relative_to(root).as_posix(), options=opts)
        teams[t['id']] = t
    return teams


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError('Tournament config must be an object')
    c = dict(config)
    allowed = {'schema_version', 'teams', 'layouts', 'repeats', 'seed', 'moves', 'game_timeout', 'record'}
    if set(c) - allowed:
        raise ValueError('Unknown config fields: ' + ', '.join(sorted(set(c) - allowed)))
    for field in ('teams', 'layouts'):
        seq = c.get(field)
        if not isinstance(seq, list) or not seq or any(not isinstance(v, str) for v in seq) or len(set(seq)) != len(seq):
            raise ValueError(field + ' must be a nonempty list of unique strings')
    if len(c['teams']) < 2 or any(not ID.fullmatch(t) for t in c['teams']):
        raise ValueError('At least two valid team IDs are required')
    for layout in c['layouts']:
        if not re.fullmatch(r'[A-Za-z0-9_]+Capture|RANDOM[1-9][0-9]*', layout):
            raise ValueError('Invalid layout: ' + layout)
    for field, default, minimum in [('schema_version', 1, 1), ('repeats', 1, 1), ('seed', 0, 0),
                                    ('moves', 1200, 1), ('game_timeout', 1800, 1)]:
        c.setdefault(field, default)
        if type(c[field]) is not int or c[field] < minimum:
            raise ValueError(field + ' must be an integer >= ' + str(minimum))
    if c['schema_version'] != 1 or c['moves'] > 1200:
        raise ValueError('Unsupported schema version or move limit above 1200')
    c.setdefault('record', False)
    if not isinstance(c['record'], bool):
        raise ValueError('record must be a boolean')
    return c


def schedule(teams, layouts, repeats, seed):
    jobs = []
    for repeat in range(repeats):
        for layout in layouts:
            for red, blue in itertools.combinations(teams, 2):
                pair = len(jobs) // 2
                game_seed = seed + repeat * len(layouts) + layouts.index(layout)
                for a, b in ((red, blue), (blue, red)):
                    jobs.append(dict(match_id=len(jobs), pair_id=pair, red=a, blue=b,
                                     layout=layout, seed=game_seed, repeat=repeat))
    return jobs


def classify(score, log, returncode):
    events = {side: dict(warnings=0, timeouts=0, crashes=0) for side in ('red', 'blue')}
    for line in log.splitlines():
        index = re.search(r'Agent (\d+)', line)
        side = ('red' if int(index.group(1)) % 2 == 0 else 'blue') if index else None
        if side:
            if 'took too long to make a move' in line:
                events[side]['warnings'] += 1
            if 'timed out' in line or 'ran out of time' in line or 'exceeded the maximum' in line:
                events[side]['timeouts'] += 1
        for side in events:
            if side.capitalize() + ' agent crashed' in line:
                events[side]['crashes'] += 1
    valid = type(score) in (int, float) and math.isfinite(score) and returncode == 0
    status = 'completed' if valid else 'error'
    if valid and any(e['crashes'] for e in events.values()):
        status = 'forfeit'
    return dict(status=status, score=score if valid else None,
                winner=('red' if score > 0 else 'blue' if score < 0 else None) if valid else None,
                events=events, returncode=returncode)
