# PacLab: three-layer capture-the-flag team.
# Framework attribution: UC Berkeley CS188, John DeNero and Dan Klein, https://ai.berkeley.edu.
#
# Layer 1 (Maze): graph, all-pairs maze distances, borders.
# Layer 2 (behaviours): escape-race attacker, region-shrinking defender, border shadowing.
# Layer 3 (Team): role assignment from score, time, invaders and cargo.
#
# Core safety idea (attacker): T(x) = earliest time any dangerous enemy ghost can stand on x.
# L(x) = latest time we may stand on x and still reach home (or a capsule) strictly ahead of every ghost.
# Moving only to cells with arrival time <= L keeps an escape route by induction.
import collections
import heapq
import os
import random

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point

INF = 10 ** 6
LOG = os.environ.get('PACLAB_LOG')
# Tunable behaviour constants; PACLAB_PARAMS (JSON) overrides them in experiments only.
PARAMS = dict(
    slack=[0],                   # extra arrival-time tolerance by carried food (index = cargo, capped)
    bank_cargo=20,               # always go home with this much
    bank_near=3, bank_near_dist=2,
    help_cargo=5,                # invader cargo that pulls the attacker into the hunt
    idle_help=1,                 # staging attacker hunts invaders
    help_near=12,                # ... only invaders within this many steps of the attacker
    stage_hold=8,                # moves to keep a staging cell
    stage_food_w=3,              # staging: weight of enemy food close behind an entry
    save_caps=8,                 # do not step on capsules while every enemy ghost is scared longer than this
    cap_guard=250,               # defender cost when an invader can reach our capsule first
    scared_attack=8,             # a ghost scared for longer than this attacks instead of defending
    pac_delay=99,                # threat delay of enemies that are invading: they keep eating (0 = off)
    shadow_depth=4,              # shadow cells at most this far from the border
    exit_w=10, exit_carry_w=6,   # hunt: cost per step of the invader's lead at its best exit (+ per carried food)
    pair_after=20,               # paired entry: after this many frozen attacker moves the defender stages too (0 = off)
    pair_cargo=3,                # ... and banks at this cargo
    end_ahead=50,                # last moves: while ahead the attacker defends ...
    end_behind=40,               # ... and while behind both attack
)
if os.environ.get('PACLAB_PARAMS') and os.environ.get('PACLAB_PARAMS_FILE', __file__) == os.path.abspath(__file__):
    import json
    PARAMS.update(json.loads(os.environ['PACLAB_PARAMS']))
STEPS = ((0, 1, Directions.NORTH), (0, -1, Directions.SOUTH),
         (1, 0, Directions.EAST), (-1, 0, Directions.WEST))


def create_team(first_index, second_index, is_red, first='PacLabAgent', second='PacLabAgent',
                num_training=0):
    team = Team()
    return [PacLabAgent(first_index, team), PacLabAgent(second_index, team)]


# ---------------------------------------------------------------- layer 1: maze

class Maze:
    def __init__(self, walls, red):
        self.width, self.height = walls.width, walls.height
        self.red = red
        self.cells = [(x, y) for x in range(self.width) for y in range(self.height) if not walls[x][y]]
        cell_set = set(self.cells)
        self.adj = {c: [(c[0] + dx, c[1] + dy) for dx, dy, _ in STEPS if (c[0] + dx, c[1] + dy) in cell_set]
                    for c in self.cells}
        self.dist = {c: self._bfs(c) for c in self.cells}
        mid = self.width // 2
        self.home_cells = [c for c in self.cells if (c[0] < mid) == red]
        self.enemy_cells = [c for c in self.cells if (c[0] < mid) != red]
        self.home_set = set(self.home_cells)
        # Border cells: home cells with an enemy-side neighbour (where we bank food / cross).
        self.home_border = [c for c in self.home_cells if any(n not in self.home_set for n in self.adj[c])]
        self.enemy_border = [c for c in self.enemy_cells if any(n in self.home_set for n in self.adj[c])]
        self.home_dist = self._multi_bfs(self.home_border)
        self.enemy_home_dist = self._multi_bfs(self.enemy_border)
        # Distances that never leave our side: a ghost walking to an entry or guard post cannot cut across.
        self.hdist = {c: self._bfs(c, self.home_set) for c in self.home_cells}

    def _bfs(self, start, allowed=None):
        dist = {start: 0}
        queue = collections.deque([start])
        while queue:
            p = queue.popleft()
            d = dist[p] + 1
            for n in self.adj[p]:
                if n not in dist and (allowed is None or n in allowed):
                    dist[n] = d
                    queue.append(n)
        return dist

    def _multi_bfs(self, starts):
        dist = {s: 0 for s in starts}
        queue = collections.deque(starts)
        while queue:
            p = queue.popleft()
            for n in self.adj[p]:
                if n not in dist:
                    dist[n] = dist[p] + 1
                    queue.append(n)
        return dist

    def d(self, a, b):
        return self.dist[a].get(b, INF)

    def is_home(self, c):
        return c in self.home_set

    @staticmethod
    def action_to(a, b):
        for dx, dy, action in STEPS:
            if (a[0] + dx, a[1] + dy) == b:
                return action
        return Directions.STOP


# ---------------------------------------------------------------- shared team state

class Team:
    """Facts shared by both agents; recomputed once per ply (keyed by timeleft)."""

    def __init__(self):
        self.maze = None
        self.indices = []
        self.roles = {}
        self.targets = {}
        self.labels = {}
        self.stage = {}
        self.hunt = {}
        self.helping = False
        self.frozen = {}
        self.defender = None
        self.last_seen = {}
        self.ply_key = None

    def setup(self, agent, state):
        if self.maze is None:
            self.maze = Maze(state.get_walls(), agent.red)
            self.indices = sorted(agent.get_team(state))
            self.enemies = sorted(agent.get_opponents(state))
            for i in self.enemies:
                self.last_seen[i] = nearest_point(state.get_initial_agent_position(i))


# ---------------------------------------------------------------- agent

class PacLabAgent(CaptureAgent):
    def __init__(self, index, team, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.team = team
        self.rng = random.Random(1009 + index)
        self.returned_seen = 0

    # Full game state: the engine hands agents the true state and observation_function
    # normally strips it; we keep it.
    def observation_function(self, game_state):
        return game_state

    def register_initial_state(self, state):
        self.red = state.is_on_red_team(self.index)
        self.register_team(self.get_team(state))
        self.start = state.get_agent_position(self.index)
        # The game ends as soon as one team has returned this many dots.
        self.to_win = (len(state.get_red_food().as_list()) + len(state.get_blue_food().as_list())) / 2 - 2
        self.team.setup(self, state)
        self.maze = self.team.maze

    def get_maze_distance(self, a, b):
        return self.maze.d(a, b)

    # ------------------------------------------------------------ perception

    def enemy_pos(self, state, i):
        pos = state.get_agent_position(i)
        if pos is not None:
            pos = nearest_point(pos)
            self.team.last_seen[i] = pos
            return pos
        return self.team.last_seen[i]

    def snapshot(self, state):
        """Per-ply facts used by every behaviour."""
        m = self.maze
        s = {}
        s['me'] = nearest_point(state.get_agent_position(self.index))
        s['my_state'] = state.get_agent_state(self.index)
        s['mate'] = [i for i in self.team.indices if i != self.index][0]
        s['mate_pos'] = nearest_point(state.get_agent_position(s['mate']))
        s['enemies'] = {}
        for i in self.team.enemies:
            st = state.get_agent_state(i)
            s['enemies'][i] = dict(pos=self.enemy_pos(state, i), pac=st.is_pacman,
                                   scared=st.scared_timer, carry=st.num_carrying)
        s['food'] = self.get_food(state).as_list()
        s['our_food'] = self.get_food_you_are_defending(state).as_list()
        s['capsules'] = self.get_capsules(state)
        s['our_caps'] = set(self.get_capsules_you_are_defending(state))
        s['moves_left'] = state.data.timeleft // 4
        s['score'] = self.get_score(state)
        s['returned'] = sum(state.get_agent_state(i).num_returned for i in self.team.indices)
        s['invaders'] = [i for i, e in s['enemies'].items() if e['pac']]
        return s

    # ------------------------------------------------------------ attacker fields

    def threat_field(self, s):
        """T(x): earliest number of our moves after which a dangerous ghost may occupy x."""
        m = self.maze
        T = {}
        # An invader keeps eating on our side instead of turning back, so it is a late threat at best.
        sources = [(m.dist[e['pos']], e['scared'], PARAMS['pac_delay'] if e['pac'] else 0)
                   for e in s['enemies'].values()]
        for c in m.enemy_cells:
            best = INF
            for dist, scared, late in sources:
                t = dist.get(c, INF) + late
                if scared:
                    t = max(t, scared)
                if t < best:
                    best = t
            T[c] = best
        return T

    def latest_field(self, s, T):
        """L(x): latest arrival time at enemy cell x that still allows a safe escape."""
        m = self.maze
        L = {}
        heap = []
        for c in m.home_border:
            for n in m.adj[c]:
                if not m.is_home(n):
                    v = T[n] - 1
                    if v > L.get(n, -INF):
                        L[n] = v
                        heapq.heappush(heap, (-v, n))
        for cap in s['capsules']:
            v = T[cap] - 1
            if v > L.get(cap, -INF):
                L[cap] = v
                heapq.heappush(heap, (-v, cap))
        done = set()
        while heap:
            negv, c = heapq.heappop(heap)
            if c in done:
                continue
            done.add(c)
            v = -negv
            for n in m.adj[c]:
                if m.is_home(n) or n in done:
                    continue
                cand = min(T[n] - 1, v - 1)
                if cand > L.get(n, -INF):
                    L[n] = cand
                    heapq.heappush(heap, (-cand, n))
        return L

    def safe_reach(self, start, L, slack=0, blocked=()):
        """BFS over cells we can stand on without losing our escape route (plus slack)."""
        m = self.maze
        reach = {start: 0}
        parent = {start: None}
        queue = collections.deque([start])
        while queue:
            p = queue.popleft()
            t = reach[p] + 1
            for n in m.adj[p]:
                if n in reach or n in blocked:
                    continue
                if m.is_home(n) or t <= L.get(n, -INF) + slack:
                    reach[n] = t
                    parent[n] = p
                    queue.append(n)
        return reach, parent

    @staticmethod
    def first_step(parent, target):
        step = target
        while parent.get(step) is not None and parent[parent[step]] is not None:
            step = parent[step]
        return step

    # ------------------------------------------------------------ behaviours

    def attack(self, state, s):
        m = self.maze
        me, carry = s['me'], s['my_state'].num_carrying
        T = self.threat_field(s)
        L = self.latest_field(s, T)
        slack_table = PARAMS['slack']
        slack = slack_table[min(carry, len(slack_table) - 1)]
        blocked = ()
        if PARAMS['save_caps']:
            ghosts = [e for e in s['enemies'].values() if not e['pac']]
            if ghosts and min(e['scared'] for e in ghosts) > PARAMS['save_caps']:
                # Enemies are already scared: stepping on another capsule would only waste it.
                blocked = set(s['capsules'])
        reach, parent = self.safe_reach(me, L, slack, blocked)
        home_now = m.is_home(me)
        to_home = m.home_dist.get(me, 0)

        foods = [f for f in s['food'] if f in reach]
        mate_target = self.team.targets.get(s['mate'])
        pair = self.team.roles.get(self.index) == 'pair'
        need_return = carry > 0 and (
            len(s['food']) <= 2 or
            (pair and carry >= PARAMS['pair_cargo']) or
            s['returned'] + carry >= self.to_win or   # banking now wins the game outright
            s['moves_left'] <= to_home + 3 or
            carry >= PARAMS['bank_cargo'] or
            (carry >= PARAMS['bank_near'] and to_home <= PARAMS['bank_near_dist']) or
            not foods)
        if need_return:
            homes = [c for c in m.home_border if c in reach]
            if homes:
                target = min(homes, key=lambda c: reach[c])
                return self.go(parent, target, 'bank')
        if foods:
            def value(f):
                v = reach[f] + 0.25 * m.home_dist.get(f, 0)
                if f == mate_target:
                    v += 6
                elif mate_target is not None and m.d(f, mate_target) <= 3:
                    v += 2
                return v
            target = min(foods, key=value)
            return self.go(parent, target, 'collect')
        caps = [c for c in s['capsules'] if c in reach]
        if caps:
            target = min(caps, key=lambda c: reach[c])
            return self.go(parent, target, 'capsule')
        if not home_now:
            homes = [c for c in m.home_border if c in reach]
            if homes:
                return self.go(parent, min(homes, key=lambda c: reach[c]), 'retreat')
            return self.evade(state, s, T, L), 'evade', None
        # Nothing safe to take: wait on our side at the entry farthest from defenders.
        held = self.team.stage.get(self.index)
        if held and held[1] > 0 and held[0] != me:
            target = held[0]
            self.team.stage[self.index] = (target, held[1] - 1)
        else:
            target = self.staging_cell(s)
            self.team.stage[self.index] = (target, PARAMS['stage_hold'])
        return self.walk(state, me, target, home_only=True), 'stage', target

    def go(self, parent, target, role):
        step = self.first_step(parent, target)
        return self.maze.action_to(self.current_pos, step), role, target

    def evade(self, state, s, T, L):
        m = self.maze
        me = s['me']
        best, best_key = Directions.STOP, None
        for action in state.get_legal_actions(self.index):
            n = self.next_cell(me, action)
            key = (L.get(n, INF if m.is_home(n) else -INF) if not m.is_home(n) else INF,
                   T.get(n, INF), -m.home_dist.get(n, 0))
            if best_key is None or key > best_key:
                best, best_key = action, key
        return best

    def staging_cell(self, s):
        """Entry to wait at: away from enemy ghosts, with enemy food close behind, spread from our mate."""
        m = self.maze
        ghosts = [e['pos'] for e in s['enemies'].values() if not e['pac'] and e['scared'] <= 2]

        food = s['food']
        mate_stage = self.team.stage.get(s['mate'], (None, 0))[0]

        def score(c):
            across = [n for n in m.adj[c] if not m.is_home(n)]
            gap = min([m.d(g, a) for g in ghosts for a in across] or [INF])
            # Opportunity: food close behind this entry, entries a ghost cannot cover at once.
            near = sum(1.0 / (1 + m.d(c, f)) for f in food if m.d(c, f) <= 8)
            spread = 0 if mate_stage is None else min(m.d(c, mate_stage), 8) * 0.3
            return min(gap, 6) + PARAMS['stage_food_w'] * near + spread - 0.25 * m.d(s['me'], c)
        cells = m.home_border
        if s['me'] in m.hdist:
            cells = [c for c in cells if c in m.hdist[s['me']]] or cells
        return max(cells, key=score)

    def defend(self, state, s, partners=()):
        """Ghost on our side. Shrink the invader's safe region, or shadow the likely attacker."""
        m = self.maze
        me = s['me']
        scared = s['my_state'].scared_timer
        invaders = [i for i in s['invaders']]
        if invaders:
            target = self.team.hunt.get(self.index)
            if target not in invaders:
                target = max(invaders, key=lambda i: (s['enemies'][i]['carry'] * 3 -
                                                       m.d(me, s['enemies'][i]['pos'])))
            inv = s['enemies'][target]['pos']
            others = [p for p in partners]
            best, best_key = Directions.STOP, None
            for action in state.get_legal_actions(self.index):
                n = self.next_cell(me, action)
                if not m.is_home(n):
                    continue
                if scared > 1 and m.d(n, inv) <= 1:
                    continue
                key = self.region_cost(inv, [n] + others, s, s['enemies'][target]['carry']) + 0.05 * m.d(n, inv)
                if scared > 1:
                    key += 0.3 * abs(m.d(n, inv) - 2)
                if best_key is None or key < best_key:
                    best, best_key = action, key
            return best, 'hunt', inv
        target = self.shadow_cell(s)
        return self.walk(state, me, target, home_only=True), 'shadow', target

    def region_cost(self, inv, guards, s, carry=0):
        """Size/value of the invader's safe region given our ghost positions (after our move)."""
        m = self.maze
        gd = [m.dist[g] for g in guards]
        if inv in guards:
            return -1000
        our_food = s['our_food_set']
        our_caps = s['our_caps']
        seen = {inv: 0}
        queue = collections.deque([inv])
        capsule = False
        food = 0
        while queue:
            p = queue.popleft()
            t = seen[p] + 1
            for n in m.adj[p]:
                if n in seen:
                    continue
                if not m.is_home(n):
                    continue
                if t < min(d.get(n, INF) for d in gd):
                    seen[n] = t
                    food += n in our_food
                    capsule = capsule or n in our_caps
                    queue.append(n)
        cost = (PARAMS['cap_guard'] if capsule else 0) + 4 * food + len(seen)
        # Graded exit cut: how far ahead the invader is at its best crossing, weighted by its cargo.
        di = m.dist[inv]
        lead = max(min(d.get(b, INF) for d in gd) - di.get(b, INF) for b in m.home_border)
        return cost + (PARAMS['exit_w'] + PARAMS['exit_carry_w'] * carry) * max(0, lead)

    def shadow_cell(self, s):
        """Cell on our side that minimises the most dangerous enemy's lead at any border entry."""
        m = self.maze
        outside = [i for i, e in s['enemies'].items() if not e['pac']]
        if not outside:
            outside = list(s['enemies'])
        target = min(outside, key=lambda i: m.home_dist.get(s['enemies'][i]['pos'], 0))
        a = s['enemies'][target]['pos']
        entries = m.home_border
        our_food = s['our_food']
        candidates = [c for c in m.home_cells if m.home_dist.get(c, INF) <= PARAMS['shadow_depth']]

        def cost(c):
            lead = max(m.d(c, b) - m.d(a, b) for b in entries)
            spread = sum(m.d(c, f) for f in our_food[:40]) / max(1, min(40, len(our_food)))
            return lead + 0.02 * spread
        return min(candidates, key=cost)

    # ------------------------------------------------------------ movement helpers

    @staticmethod
    def next_cell(p, action):
        for dx, dy, a in STEPS:
            if a == action:
                return (p[0] + dx, p[1] + dy)
        return p

    def walk(self, state, start, target, home_only=False):
        """Shortest-path step toward target (optionally never leaving our side)."""
        m = self.maze
        legal = state.get_legal_actions(self.index)
        dist = m.dist[target]
        if home_only and start in m.hdist.get(target, ()):
            dist = m.hdist[target]   # the full-maze path may cross the border; greedy steps on it get stuck
        best, best_d = Directions.STOP, dist.get(start, INF)
        for action in legal:
            n = self.next_cell(start, action)
            if n == start:
                continue
            if home_only and not m.is_home(n):
                continue
            d = dist.get(n, INF)
            if d < best_d:
                best, best_d = action, d
        return best

    # ------------------------------------------------------------ layer 3: team strategy

    def assign_roles(self, state, s):
        """Decide attacker/defender for both agents once per ply."""
        team = self.team
        key = state.data.timeleft
        if team.ply_key == key:
            return team.roles
        team.ply_key = key
        m = self.maze
        agents = {}
        for i in team.indices:
            st = state.get_agent_state(i)
            agents[i] = dict(pos=nearest_point(st.get_position()), pac=st.is_pacman,
                             carry=st.num_carrying, scared=st.scared_timer)
        invaders = s['invaders']
        inv_carry = sum(s['enemies'][i]['carry'] for i in invaders)
        anchor_pos = [s['enemies'][i]['pos'] for i in invaders] or [self.shadow_cell(s)]

        def defend_cost(i):
            a = agents[i]
            c = min(m.d(a['pos'], p) for p in anchor_pos)
            if a['carry'] > 0:
                c += 4 + 2 * a['carry']
            if a['scared'] > 3:
                c += 15
            return c

        current = team.defender if team.defender in agents else None
        best = min(agents, key=defend_cost)
        if current is None or defend_cost(best) + 6 < defend_cost(current):
            current = best
        team.defender = current
        roles = {i: 'attack' for i in agents}
        roles[current] = 'defend'
        other = [i for i in agents if i != current][0]

        lead = s['score']
        moves_left = s['moves_left']
        # Attack opportunity of an agent = safe food it can reach right now (state based, so no flapping).
        T = self.threat_field(s)
        L = self.latest_field(s, T)
        food = s['food']

        def opportunity(i):
            reach, _ = self.safe_reach(agents[i]['pos'], L)
            return sum(1 for f in food if f in reach)
        opp_other = opportunity(other)
        if not invaders:
            team.helping = False
        else:
            o = agents[other]
            near = min(m.d(o['pos'], s['enemies'][i]['pos']) for i in invaders)
            urgent = inv_carry >= PARAMS['help_cargo']
            if team.helping:
                if opp_other >= 3 and not urgent:
                    team.helping = False
            elif (urgent or (PARAMS['idle_help'] and opp_other == 0)) and o['carry'] <= 2 and \
                    near <= PARAMS['help_near']:
                team.helping = True
        if team.helping:
            roles[other] = 'defend'
        # Paired entry: in a border standoff (nobody inside, our attacker frozen) the defender also waits at an
        # entry, so a single enemy ghost cannot cover both of us; it returns on any invader.
        if PARAMS['pair_after'] and not invaders and roles[current] == 'defend' and roles[other] == 'attack' and \
                team.frozen.get(other, 0) >= PARAMS['pair_after'] and \
                not (moves_left < PARAMS['end_ahead'] and lead > 0):
            roles[current] = 'pair'
        for i in agents:
            if agents[i]['scared'] > PARAMS['scared_attack']:
                roles[i] = 'attack'
        if moves_left < PARAMS['end_ahead'] and lead > 0 and agents[other]['carry'] == 0:
            roles[other] = 'defend'
        if moves_left < PARAMS['end_behind'] and lead < 0:
            roles[current] = 'attack'
        team.roles = roles
        defenders = [i for i in roles if roles[i] == 'defend']
        team.hunt = {}
        if len(defenders) == 2 and invaders:
            if len(invaders) == 1:
                # Pincer: both ghosts squeeze the same invader (one ghost cannot catch in a loop).
                target = max(invaders, key=lambda i: (s['enemies'][i]['carry'] * 3 - min(
                    m.d(agents[d]['pos'], s['enemies'][i]['pos']) for d in defenders)))
                team.hunt = {d: target for d in defenders}
            else:
                a, b = defenders
                x, y = invaders
                px, py = s['enemies'][x]['pos'], s['enemies'][y]['pos']
                if m.d(agents[a]['pos'], px) + m.d(agents[b]['pos'], py) <= \
                        m.d(agents[a]['pos'], py) + m.d(agents[b]['pos'], px):
                    team.hunt = {a: x, b: y}
                else:
                    team.hunt = {a: y, b: x}
        return roles

    # ------------------------------------------------------------ main

    def choose_action(self, state):
        try:
            return self._choose(state)
        except Exception:
            if os.environ.get('PACLAB_STRICT'):
                raise
            legal = state.get_legal_actions(self.index)
            moving = [a for a in legal if a != Directions.STOP]
            return self.rng.choice(moving or legal)

    def _choose(self, state):
        s = self.snapshot(state)
        s['our_food_set'] = set(s['our_food'])
        self.current_pos = s['me']
        roles = self.assign_roles(state, s)
        role = roles.get(self.index, 'attack')
        m = self.maze
        if role == 'defend' and m.is_home(s['me']):
            partners = [s['mate_pos']] if roles.get(s['mate']) == 'defend' and m.is_home(s['mate_pos']) else []
            action, label, target = self.defend(state, s, partners)
        elif role == 'defend':
            # Defender still on enemy side: get home safely first.
            T = self.threat_field(s)
            L = self.latest_field(s, T)
            reach, parent = self.safe_reach(s['me'], L)
            homes = [c for c in m.home_border if c in reach]
            if homes:
                action, label, target = self.go(parent, min(homes, key=lambda c: reach[c]), 'return')
            else:
                action, label, target = self.evade(state, s, T, L), 'evade', None
        else:
            action, label, target = self.attack(state, s)
            # Any ghost of ours eats an adjacent invader on the way.
            if m.is_home(s['me']) and s['my_state'].scared_timer == 0:
                for i in s['invaders']:
                    p = s['enemies'][i]['pos']
                    if m.d(s['me'], p) == 1:
                        action, label, target = m.action_to(s['me'], p), 'kill', p
        self.team.targets[self.index] = target
        self.team.labels[self.index] = label
        frozen = self.team.frozen
        st = s['my_state']
        if st.num_carrying > 0 or st.num_returned > self.returned_seen or self.current_pos == self.start:
            frozen[self.index] = 0
        elif role == 'attack' and not st.is_pacman:
            frozen[self.index] = frozen.get(self.index, 0) + 1
        self.returned_seen = st.num_returned
        if action not in state.get_legal_actions(self.index):
            action = Directions.STOP
        if LOG:
            with open(LOG, 'a') as f:
                ghosts = sum(1 for e in s['enemies'].values() if not e['pac'])
                f.write(f"{1200 - state.data.timeleft} {self.index} {role} {label} {s['me']} {target} {action} "
                        f"g{ghosts} c{s['my_state'].num_carrying}\n")
        return action
