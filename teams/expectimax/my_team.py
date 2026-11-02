# PacLab original strategy implementation for the UPF course framework.
# Framework attribution: UC Berkeley CS188, John DeNero and Dan Klein, https://ai.berkeley.edu.
# Do not distribute course solutions outside the permitted private team repository.
import collections
import heapq
import json
import math
import os
import random
import time

from contest.capture_agents import CaptureAgent
from contest.game import Directions, Configuration
from contest.util import nearest_point

VECTORS = {Directions.NORTH: (0, 1), Directions.SOUTH: (0, -1),
           Directions.EAST: (1, 0), Directions.WEST: (-1, 0), Directions.STOP: (0, 0)}
FEATURE_SCHEMA = ['bias', 'goal_progress', 'food_eaten', 'food_returned', 'danger_or_death',
                  'dangerous_dead_end', 'stop', 'reverse', 'capsule_eaten', 'score_change',
                  'recent_visits', 'defending_home', 'carried_food', 'home_burden',
                  'invader_captured', 'safe_exits', 'teammate_conflict', 'return_deadline']
FEATURE_SCHEMA_VERSION = 2
DEFAULT_WEIGHTS = [0.0, 18.0, 12.0, 20.0, -28.0, -6.0, -2.0, -1.0, 9.0, 3.0, -2.0, 0.8,
                   2.0, -0.8, 20.0, 1.0, -3.0, -12.0]


class SharedTeam:
    def __init__(self, inference='exact'):
        self.inference = inference
        self.ready = False
        self.beliefs = {}
        self.targets = {}
        self.defending = {}
        self.last_index = None
        self.last_time = None
        self.last_food = set()
        self.losses = []
        self.rng = random.Random(271828)
        self.behaviour = {}
        self.previous_visible = {}
        self.previous_public = {}
        self.spawn_positions = {}

    def initialise(self, agent, state):
        if self.ready:
            return
        walls = state.get_walls()
        self.width, self.height = walls.width, walls.height
        self.cells = [(x, y) for x in range(walls.width) for y in range(walls.height) if not walls[x][y]]
        cell_set = set(self.cells)
        self.adj = {p: [(p[0] + dx, p[1] + dy) for dx, dy in VECTORS.values()
                        if (dx or dy) and (p[0] + dx, p[1] + dy) in cell_set] for p in self.cells}
        bx = walls.width // 2 - 1 if agent.red else walls.width // 2
        self.border = [p for p in self.cells if p[0] == bx and
                       (p[0] + (1 if agent.red else -1), p[1]) in cell_set]
        if not self.border:
            self.border = [state.get_agent_position(agent.index)]
        self.depth = {}
        degrees = {p: len(n) for p, n in self.adj.items()}
        queue = collections.deque(p for p in self.cells if degrees[p] == 1 and p not in self.border)
        while queue:
            p = queue.popleft()
            self.depth[p] = 1 + max([self.depth.get(n, 0) for n in self.adj[p]] or [0])
            for n in self.adj[p]:
                degrees[n] -= 1
                if degrees[n] == 1 and n not in self.depth and n not in self.border:
                    queue.append(n)
        removed = set(self.depth)
        distance_to_exit = {p: 0 for p in self.cells if p not in removed}
        frontier = collections.deque(distance_to_exit)
        while frontier:
            p = frontier.popleft()
            for q in self.adj[p]:
                if q not in distance_to_exit:
                    distance_to_exit[q] = distance_to_exit[p] + 1
                    frontier.append(q)
        self.depth = {p: distance_to_exit.get(p, 1) for p in removed}
        self.beliefs = {i: {nearest_point(state.get_agent_position(i)): 1.0} for i in agent.get_opponents(state)}
        self.spawn_positions = {i: nearest_point(state.get_initial_agent_position(i)) for i in self.beliefs}
        self.previous_public = {i: (state.get_agent_state(i).is_pacman,
                                    state.get_agent_state(i).num_carrying,
                                    state.get_agent_state(i).num_returned) for i in self.beliefs}
        self.behaviour = {i: {'moves': 0, 'chases': 0} for i in self.beliefs}
        self.initial_time = state.data.timeleft
        self.last_food = set(agent.get_food_you_are_defending(state).as_list())
        self.ready = True

    def observe(self, agent, state):
        tick = state.data.timeleft
        food = set(agent.get_food_you_are_defending(state).as_list())
        self.losses = list(self.last_food - food)
        elapsed = max(0, (self.initial_time if self.last_time is None else self.last_time) - tick)
        for i in self.beliefs:
            b = self.beliefs[i]
            actor = state.get_agent_state(i)
            previous = self.previous_public.get(i)
            captured = (previous is not None and previous[0] and previous[1] > 0 and
                        not actor.is_pacman and actor.num_carrying == 0 and
                        actor.num_returned == previous[2])
            self.previous_public[i] = (actor.is_pacman, actor.num_carrying, actor.num_returned)
            if captured:
                # Public load loss without a deposit proves capture. Its exact
                # time is unknown: allow every elapsed enemy turn after spawn.
                b = {self.spawn_positions[i]: 1.0}
            if elapsed:
                first = (agent.index - elapsed) % 4 if self.last_index is None else self.last_index
                steps = sum((first + n) % 4 == i for n in range(elapsed))
                for _ in range(steps):
                    moved = collections.defaultdict(float)
                    if self.inference == 'particle' and not captured:
                        samples = self.rng.choices(list(b), weights=list(b.values()), k=160)
                        for p in samples:
                            moved[self.rng.choice(self.adj[p] + [p])] += 1.0 / len(samples)
                    else:
                        for p, mass in b.items():
                            destinations = self.adj[p] + [p]
                            for q in destinations:
                                moved[q] += mass / len(destinations)
                    b = dict(moved)
            visible = state.get_agent_state(i).get_position()
            if visible is not None:
                previous = self.previous_visible.get(i)
                if previous and previous[0] != visible and not state.get_agent_state(i).is_pacman:
                    old_pos, target = previous
                    self.behaviour[i]['moves'] += 1
                    self.behaviour[i]['chases'] += int(agent.distance(visible, target) < agent.distance(old_pos, target))
                self.previous_visible[i] = (visible, state.get_agent_position(agent.index))
                self.beliefs[i] = {nearest_point(visible): 1.0}
                continue
            my_pos = state.get_agent_position(agent.index)
            own = [state.get_agent_position(j) for j in agent.get_team(state)]
            noisy = state.get_agent_distances()[i]
            is_pacman = state.get_agent_state(i).is_pacman
            eligible = [p for p in self.cells if ((p[0] < self.width // 2) == agent.red) == is_pacman
                        and all(abs(p[0] - q[0]) + abs(p[1] - q[1]) > 5 for q in own if q is not None)
                        and state.get_distance_prob(abs(p[0] - my_pos[0]) + abs(p[1] - my_pos[1]), noisy) > 0]
            allowed = set(eligible)
            weighted = {p: mass for p, mass in b.items() if p in allowed}
            if not weighted:
                weighted = {p: 1.0 for p in eligible or self.cells}
            if is_pacman and self.losses:
                weighted = {p: mass * (4.0 if any(abs(p[0] - q[0]) + abs(p[1] - q[1]) <= 2 for q in self.losses) else 1.0)
                            for p, mass in weighted.items()}
            total = sum(weighted.values())
            self.beliefs[i] = {p: mass / total for p, mass in weighted.items()}
        self.last_food = food
        self.last_time, self.last_index = tick, agent.index


class PacLabAgent(CaptureAgent):
    mode = 'astar'

    def __init__(self, index, shared, slot, budget=0.07):
        super().__init__(index)
        self.shared, self.slot, self.budget = shared, slot, float(budget)
        self.rng = random.Random(314159 + index)
        self.history = collections.deque(maxlen=12)
        self.weights = list(DEFAULT_WEIGHTS)
        self.model = None
        self.action_cache = {}
        self.recovery_goal = None
        self.recovery_until = 0

    def register_initial_state(self, state):
        super().register_initial_state(state)
        self.start = state.get_agent_position(self.index)
        self.shared.initialise(self, state)
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.json')
        if self.mode in ('qlearning', 'evolution', 'neural'):
            if not os.path.isfile(model_path):
                raise ValueError('Learned team requires its trained model.json')
            with open(model_path) as f:
                model = json.load(f)
            if model['kind'] != self.mode or model['feature_count'] != len(DEFAULT_WEIGHTS):
                raise ValueError('Model type/feature schema mismatch')
            if model.get('feature_schema') != FEATURE_SCHEMA:
                raise ValueError('Model feature names/order mismatch')
            def finite(value):
                return type(value) in (int, float) and math.isfinite(value)
            if self.mode != 'neural':
                if len(model.get('weights', [])) != len(DEFAULT_WEIGHTS) or not all(finite(v) for v in model['weights']):
                    raise ValueError('Invalid learned linear weights')
            else:
                hidden = len(model.get('b1', []))
                if not hidden or len(model.get('w1', [])) != hidden or len(model.get('w2', [])) != hidden or any(len(row) != len(DEFAULT_WEIGHTS) for row in model['w1']) or not all(finite(v) for row in model['w1'] for v in row) or not all(finite(v) for v in model['b1'] + model['w2'] + [model['b2']]):
                    raise ValueError('Invalid neural model dimensions or values')
            self.model = model
            if self.mode != 'neural':
                self.weights = model['weights']

    def distance(self, a, b):
        if a is None or b is None:
            return 9999
        return self.get_maze_distance(a, b)

    def risk_at(self, position, state=None, index=None):
        state = self.current if state is None else state
        index = self.index if index is None else index
        current = state is self.current
        p = nearest_point(position)
        key = (p, index)
        if current and key in self.risk_cache:
            return self.risk_cache[key]
        value = 0.0
        our = state.get_agent_state(index)
        for i, belief in self.shared.beliefs.items():
            enemy = state.get_agent_state(i)
            dangerous = (not enemy.is_pacman and enemy.scared_timer <= 2 and self.on_enemy_side(p)) or (
                enemy.is_pacman and our.scared_timer > 0 and not self.on_enemy_side(p))
            if not dangerous:
                continue
            visible = enemy.get_position()
            if visible is not None:
                belief = {visible: 1.0}
            # Manhattan distance lower-bounds maze distance: distant mass contributes exactly zero.
            value += sum(prob * max(0.0, 1.0 - self.distance(p, q) / 5.0) for q, prob in belief.items()
                         if abs(p[0] - q[0]) + abs(p[1] - q[1]) < 5)
        if current:
            self.risk_cache[key] = value
        return value

    def on_enemy_side(self, p):
        return (p[0] < self.shared.width // 2) != self.red

    def carrying_limit(self, state, index=None):
        index = self.index if index is None else index
        danger = self.risk_at(state.get_agent_state(index).get_position(), state=state, index=index)
        limit = 3 if danger >= 0.15 else 5 if danger >= 0.05 else 8
        if self.get_score(state) >= 4 and state.data.timeleft < 400:
            limit = min(limit, 4)
        return limit

    def defender_assignments(self, state, invaders):
        defenders = [i for i in sorted(self.get_team(state)) if
                     not state.get_agent_state(i).is_pacman and state.get_agent_state(i).scared_timer == 0]
        threats = [(i, max(self.shared.beliefs[i], key=self.shared.beliefs[i].get)) for i in invaders]
        result = {}
        identities = {}
        while defenders and threats:
            _, defender, invader, target = min(
                (self.distance(state.get_agent_state(d).get_position(), p) -
                 min(5, state.get_agent_state(i).num_carrying), d, i, p)
                for d in defenders for i, p in threats)
            result[defender] = target
            identities[defender] = invader
            defenders.remove(defender)
            threats = [(i, p) for i, p in threats if i != invader]
        self.shared.defending = identities
        return result

    def food_assignments(self, state, food):
        """Choose two targets jointly instead of relying on decision call order."""
        indices = [i for i in sorted(self.get_team(state)) if
                   state.get_agent_state(i).num_carrying < self.carrying_limit(state, i)]
        if len(indices) != 2 or len(food) < 2:
            return {}
        costs = {}
        candidates = {}
        for index in indices:
            position = state.get_agent_state(index).get_position()
            previous = self.shared.targets.get(index)
            costs[index] = {p: self.distance(position, p) + 18 * self.risk_at(p) +
                            2 * self.shared.depth.get(p, 0) * self.risk_at(p) -
                            (1.5 if p == previous else 0) for p in food}
            candidates[index] = sorted(food, key=lambda p: (costs[index][p], p))[:10]
        first, second = indices
        pairs = [(costs[first][a] + costs[second][b] +
                  max(0, 5 - self.distance(a, b)) * 1.5, a, b)
                 for a in candidates[first] for b in candidates[second] if a != b]
        if not pairs:
            return {}
        _, a, b = min(pairs)
        return {first: a, second: b}

    def choose_goal(self, state):
        mine = state.get_agent_state(self.index)
        position = mine.get_position()
        food = self.get_food(state).as_list()
        invaders = [i for i in self.get_opponents(state) if state.get_agent_state(i).is_pacman]
        border = min(self.shared.border, key=lambda p: self.distance(position, p) + 12 * self.risk_at(p))
        home_dist = self.distance(position, border)
        carrying_limit = self.carrying_limit(state)
        returning = mine.num_carrying >= carrying_limit or (mine.num_carrying > 0 and
                    (self.risk_at(position) > 0.28 or state.data.timeleft // 4 <= home_dist + 5 or len(food) <= 2))
        if returning:
            return 'return', border
        if invaders:
            if not mine.is_pacman and mine.scared_timer > 0 and self.risk_at(position) > 0.1:
                safe = min(self.shared.cells, key=lambda p: self.risk_at(p) * 30 + self.distance(position, p)
                           + (20 if self.on_enemy_side(p) else 0))
                return 'escape', safe
            assigned = self.defender_assignments(state, invaders)
            if self.index in assigned:
                return 'defend', assigned[self.index]
        if mine.scared_timer <= 0 and self.slot == 1 and self.get_score(state) >= 4 and state.data.timeleft < 400:
            defending = self.get_food_you_are_defending(state).as_list()
            target = min(self.shared.border, key=lambda p: sum(self.distance(p, q) for q in defending) / max(1, len(defending)))
            return 'guard', target
        capsules = self.get_capsules(state)
        if capsules and self.risk_at(position) > 0.15:
            target = min(capsules, key=lambda p: self.distance(position, p))
            if self.distance(position, target) <= home_dist + 3:
                return 'capsule', target
        if not food:
            return 'return', border
        assignments = self.food_assignments(state, food)
        if self.index in assignments:
            return 'collect', assignments[self.index]
        other_targets = [p for i, p in self.shared.targets.items() if i != self.index]
        target = min(food, key=lambda p: self.distance(position, p) + 18 * self.risk_at(p) +
                     2 * self.shared.depth.get(p, 0) * self.risk_at(p) +
                     (7 if p in other_targets else 0) +
                     0.12 * self.distance(p, border) -
                     0.15 * sum(abs(p[0] - q[0]) + abs(p[1] - q[1]) <= 2 for q in food))
        return 'collect', target

    def successor(self, state, action):
        return state.generate_successor(self.index, action)

    def features(self, state, action):
        successor = self.successor(state, action)
        a = state.get_agent_state(self.index)
        b = successor.get_agent_state(self.index)
        pos = b.get_position()
        scale = max(self.shared.width, self.shared.height)
        before_dist = self.distance(a.get_position(), self.goal)
        after_dist = self.distance(pos, self.goal)
        risk = self.risk_at(pos, state=successor)
        returned = b.num_returned - a.num_returned
        food = max(0, self.get_food(state).count() - self.get_food(successor).count())
        capsule = len(self.get_capsules(state)) - len(self.get_capsules(successor))
        died = self.distance(pos, self.start) == 0 and self.distance(a.get_position(), self.start) > 2
        home = min(self.distance(pos, p) for p in self.shared.border)
        captured = sum(state.get_agent_position(i) is not None and state.get_agent_state(i).is_pacman and
                       not successor.get_agent_state(i).is_pacman for i in self.get_opponents(state))
        exits = self.shared.adj[nearest_point(pos)]
        safe_exits = sum(self.risk_at(q, state=successor) < 0.15 for q in exits) / 4.0
        teammates = [i for i in self.get_team(state) if i != self.index]
        conflict = any(self.distance(pos, successor.get_agent_position(i)) <= 1 and
                       (pos == successor.get_agent_position(i) or
                        self.shared.targets.get(i) == self.goal) for i in teammates)
        return [1.0, (before_dist - after_dist) / scale,
                food * (1 if self.role in ('collect', 'capsule') else 0),
                returned / 5.0, risk + (2.0 if died else 0.0),
                risk * self.shared.depth.get(nearest_point(pos), 0) / 5.0,
                float(action == Directions.STOP),
                float(action == Directions.REVERSE[a.configuration.direction]),
                capsule * (0.3 + self.risk_at(a.get_position())),
                (self.get_score(successor) - self.get_score(state)) / 5.0,
                self.history.count(nearest_point(pos)) / 4.0,
                (1 if self.role == 'defend' and not b.is_pacman else 0),
                b.num_carrying / 10.0, b.num_carrying * home / scale,
                float(captured), safe_exits, float(conflict),
                float(b.num_carrying > 0 and successor.data.timeleft // 4 <= home + 3)]

    def linear(self, state, action):
        features = self.features(state, action)
        return sum(x * w for x, w in zip(features, self.weights))

    def greedy(self, state):
        actions = state.get_legal_actions(self.index)
        if self.role == 'guard' and self.distance(state.get_agent_position(self.index), self.goal) == 0 and Directions.STOP in actions:
            return Directions.STOP
        return max(actions, key=lambda a: (self.linear(state, a), a != Directions.STOP, a))

    def astar(self, state, target=None):
        target = target or self.goal
        start = nearest_point(state.get_agent_position(self.index))
        legal = state.get_legal_actions(self.index)
        if start == target:
            return self.greedy(state)
        queue = [(self.distance(start, target), 0.0, start, None)]
        best = {start: 0.0}
        while queue and time.monotonic() < self.deadline:
            _, cost, pos, first = heapq.heappop(queue)
            if cost != best.get(pos):
                continue
            if pos == target:
                return first if first in legal else self.greedy(state)
            for q in self.shared.adj[pos]:
                action = next(a for a, (dx, dy) in VECTORS.items() if (dx, dy) == (q[0] - pos[0], q[1] - pos[1]))
                if first is None and action not in legal:
                    continue
                risk = self.risk_at(q)
                step = 1 + 12 * risk + risk * self.shared.depth.get(q, 0) + (0.5 if q in self.history else 0)
                new_cost = cost + step
                if new_cost < best.get(q, float('inf')):
                    best[q] = new_cost
                    heapq.heappush(queue, (new_cost + self.distance(q, target), new_cost, q, first or action))
        return self.greedy(state)

    def choose_action(self, state):
        self.deadline = time.monotonic() + min(0.35, max(0.01, self.budget))
        self.current = state
        self.risk_cache = {}
        self.shared.observe(self, state)
        self.role, self.goal = self.choose_goal(state)
        position = nearest_point(state.get_agent_state(self.index).get_position())
        if self.role == 'collect':
            if self.recovery_goal is not None and state.data.timeleft > self.recovery_until and position != self.recovery_goal:
                self.role, self.goal = 'recover', self.recovery_goal
            elif self.history.count(position) >= 4 and len(set(self.history)) <= 3:
                alternatives = [p for p in self.shared.cells if p not in self.history and
                                2 <= self.distance(position, p) <= 6]
                if alternatives:
                    self.recovery_goal = min(alternatives, key=lambda p: (
                        20 * self.risk_at(p) + 0.15 * self.shared.depth.get(p, 0) +
                        0.2 * self.distance(position, p) - 0.6 * len(self.shared.adj[p]), p))
                    self.recovery_until = state.data.timeleft - 16
                    self.role, self.goal = 'recover', self.recovery_goal
        self.shared.targets[self.index] = self.goal
        self.history.append(position)
        if self.mode == 'fsm':
            action = self.greedy(state)
        elif self.mode == 'astar':
            action = self.astar(state)
        else:
            action = self.decide(state)
        return action if action in state.get_legal_actions(self.index) else self.greedy(state)

"""Distinct decision algorithms layered over shared observation and safety support."""
import collections
import math
import random
import time


class SearchDeadline(Exception):
    pass


class StrategyAgent(PacLabAgent):
    def search_position(self, state, index):
        return state.get_agent_state(index).get_position()

    def terminal(self, state):
        return state.is_over() or state.data.timeleft <= 0

    def assigned_invader(self, state):
        # Bind the shared target to a public or inferred identity in the observation.
        # Keep that identity throughout a projected tree instead of stealing a closer threat.
        assigned = self.shared.defending.get(self.index)
        if assigned is not None and self.current.get_agent_state(assigned).is_pacman:
            enemy = state.get_agent_state(assigned)
            return enemy if enemy.is_pacman and enemy.get_position() is not None else None
        candidates = []
        for index in self.get_opponents(self.current):
            if not self.current.get_agent_state(index).is_pacman:
                continue
            position = self.search_position(self.current, index)
            if position is None:
                belief = self.shared.beliefs.get(index, {})
                position = max(belief, key=belief.get) if belief else None
            if position is not None:
                candidates.append((self.distance(position, self.goal), index))
        if not candidates:
            return None
        index = min(candidates)[1]
        enemy = state.get_agent_state(index)
        return enemy if enemy.is_pacman and enemy.get_position() is not None else None

    def projected_goal(self, state):
        mine = state.get_agent_state(self.index)
        pos = mine.get_position()
        food = self.get_food(state).as_list()
        if mine.num_carrying and (self.role == 'return' or mine.num_carrying >= self.carrying_limit(state) or
                len(food) <= 2 or state.data.timeleft // state.get_num_agents() <=
                min(self.distance(pos, p) for p in self.shared.border) + 3):
            return min(self.shared.border, key=lambda p: self.distance(pos, p) + 10 * self.risk_at(p, state))
        if self.role == 'defend':
            enemy = self.assigned_invader(state)
            return enemy.get_position() if enemy is not None else self.goal
        if (self.role == 'collect' and self.goal not in food) or (
                self.role == 'capsule' and self.goal not in self.get_capsules(state)):
            if food:
                return min(food, key=lambda p: self.distance(pos, p) + 12 * self.risk_at(p, state))
        return self.goal

    def node_value(self, state):
        mine = state.get_agent_state(self.index)
        pos = mine.get_position()
        foods = self.get_food(state).as_list()
        home = min(self.distance(pos, p) for p in self.shared.border)
        score = self.get_score(state)
        value = 100 * score + 5 * mine.num_carrying - 3 * len(foods)
        value -= 1.2 * self.distance(pos, self.projected_goal(state))
        value -= 0.3 * mine.num_carrying * home
        value -= (30 + 6 * mine.num_carrying) * self.risk_at(pos, state=state)
        value -= 12 * len(self.get_capsules(state))
        origin = self.current.get_agent_state(self.index)
        if pos == self.start and self.distance(origin.get_position(), self.start) > 2:
            value -= 35 + 8 * max(0, origin.num_carrying - mine.num_carrying)
        if self.role == 'defend':
            invaders = [state.get_agent_state(i) for i in self.get_opponents(state)
                        if state.get_agent_state(i).is_pacman]
            value -= 24 * len(invaders) + 8 * sum(i.num_carrying for i in invaders)
            value -= 12 * int(mine.is_pacman)
        if self.terminal(state):
            value += 10000 * (1 if score > 0 else -1 if score < 0 else 0)
        return value

    def opponent_actions(self, state, index):
        actions = state.get_legal_actions(index)
        opponent = state.get_agent_state(index)
        own = state.get_agent_state(self.index)
        target = own.get_position()
        food = state.get_blue_food().as_list() if state.is_on_red_team(index) else state.get_red_food().as_list()
        home_x = self.shared.width // 2 - int(state.is_on_red_team(index))
        home = [(home_x, y) for y in range(self.shared.height) if (home_x, y) in self.shared.cells and
                (home_x + (1 if state.is_on_red_team(index) else -1), y) in self.shared.cells]
        values = []
        for action in actions:
            successor = state.generate_successor(index, action)
            future = successor.get_agent_state(index)
            pos = future.get_position()
            value = -0.2 * int(action == Directions.STOP)
            if opponent.is_pacman:
                returning = opponent.num_carrying >= 5 or len(food) <= 2 or (
                    opponent.num_carrying > 0 and state.data.timeleft // state.get_num_agents() <=
                    min([self.distance(opponent.get_position(), p) for p in home] or [0]) + 3)
                objectives = home if returning else food
                value -= min([self.distance(pos, p) for p in objectives] or [0])
                value += 15 * (future.num_returned - opponent.num_returned)
                threats = [state.get_agent_state(i) for i in self.get_team(state)
                           if self.search_position(state, i) is not None and
                           not state.get_agent_state(i).is_pacman and state.get_agent_state(i).scared_timer <= 1]
                value -= 8 * sum(max(0, 2 - self.distance(pos, g.get_position())) for g in threats)
            elif own.is_pacman:
                value += (1 if opponent.scared_timer > 1 else -1) * self.distance(pos, target)
            else:
                value -= min([self.distance(pos, p) for p in home] or [0]) * 0.3
            if pos == state.get_initial_agent_position(index) and self.distance(opponent.get_position(), pos) > 2:
                value -= 20 + 5 * opponent.num_carrying
            values.append(value)
        maximum = max(values)
        weights = [math.exp(max(-12, v - maximum)) for v in values]
        total = sum(weights)
        # Uncertainty shrinks with observed evidence but always retains exploratory mass.
        evidence = self.shared.behaviour.get(index, {}).get('moves', 0)
        uniform = 0.1 + 0.3 / (1 + evidence / 8.0)
        return [(a, (1 - uniform) * w / total + uniform / len(actions)) for a, w in zip(actions, weights)]

    def round_actors(self, state):
        return [(self.index + offset) % state.get_num_agents() for offset in range(state.get_num_agents())]

    def passive_action(self, state, index):
        legal = state.get_legal_actions(index)
        # Predict only the last shared teammate objective; never call its policy or inference.
        target = self.shared.targets.get(index) if index in self.get_team(state) else None
        if target is None or target not in self.shared.cells:
            return Directions.STOP if Directions.STOP in legal else legal[0]
        mine = state.get_agent_state(index)
        if mine.num_carrying >= self.carrying_limit(state, index):
            target = min(self.shared.border, key=lambda p: self.distance(mine.get_position(), p) +
                         12 * self.risk_at(p, state=state, index=index))
        choices = {}
        for action in legal:
            future = state.generate_successor(index, action)
            projected = future.get_agent_state(index)
            pos = projected.get_position()
            choices[action] = -self.distance(pos, target) - 12 * self.risk_at(pos, state=future, index=index)
            choices[action] += 15 * (projected.num_returned - mine.num_returned)
            if pos == state.get_initial_agent_position(index) and self.distance(mine.get_position(), pos) > 2:
                choices[action] -= 30 + 8 * mine.num_carrying
        return max(choices, key=lambda a: (choices[a], a == Directions.STOP if mine.get_position() == target else a != Directions.STOP, a))

    def action_values(self, state):
        return {a: self.node_value(state.generate_successor(self.index, a))
                for a in state.get_legal_actions(self.index)}

    def prepare_search_goal(self, state):
        """Give a stalled, empty home collector a persistent safe food objective.

        A short tactical horizon cannot value the initial steps away from a camped
        nearest target. Retarget only after observed stagnation, and only when a
        distinct safe food plus its deposit route fits the remaining game time.
        """
        if state is not self.current or time.monotonic() >= self.deadline:
            return
        mine = state.get_agent_state(self.index)
        if self.role != 'collect' or mine.num_carrying or mine.is_pacman:
            self.stalemate_goal = None
            return
        position = mine.get_position()
        foods = self.get_food(state).as_list()
        claimed = {p for i, p in self.shared.targets.items() if i != self.index}
        turns_left = state.data.timeleft // state.get_num_agents()
        previous = getattr(self, 'stalemate_goal', None)
        if previous in foods and previous not in claimed and state.data.timeleft > getattr(self, 'stalemate_until', 0):
            home = min(self.distance(previous, p) for p in self.shared.border)
            if self.risk_at(previous, state) <= 0.3 and self.distance(position, previous) + home + 2 <= turns_left:
                self.goal = previous
                self.shared.targets[self.index] = previous
                return
        self.stalemate_goal = None
        if self.history.count(nearest_point(position)) < 4 or self.risk_at(self.goal, state) <= 0.3:
            return
        campers = [i for i in self.get_opponents(state) if self.search_position(state, i) is not None and
                   not state.get_agent_state(i).is_pacman and state.get_agent_state(i).scared_timer <= 2 and
                   self.distance(position, self.search_position(state, i)) <= 5]
        if not campers:
            return
        choices = []
        for food in foods:
            if time.monotonic() >= self.deadline:
                break
            if food == self.goal or food in claimed:
                continue
            danger = self.risk_at(food, state)
            if danger > 0.2:
                continue
            distance = self.distance(position, food)
            home = min(self.distance(food, p) for p in self.shared.border)
            if distance + home + 2 <= turns_left:
                choices.append((distance + 12 * danger, food, distance))
        if choices:
            _, goal, distance = min(choices)
            self.stalemate_goal = self.goal = goal
            self.stalemate_until = state.data.timeleft - state.get_num_agents() * (distance + 4)
            self.shared.targets[self.index] = goal

    def adversarial(self, state, stochastic=False):
        self.prepare_search_goal(state)
        values = self.action_values(state)
        self.last_search_depth, self.last_search_values = 0, values
        best_action = max(values, key=lambda a: (values[a], a != Directions.STOP, a))
        pos = self.search_position(state, self.index)
        active = {i for i in self.get_opponents(state) if self.search_position(state, i) is not None and
                  self.distance(pos, self.search_position(state, i)) <= 8}
        if not active or time.monotonic() >= self.deadline:
            return best_action if time.monotonic() >= self.deadline else self.astar(state)
        actors = self.round_actors(state)

        def visit(node, ply, depth, alpha, beta):
            if time.monotonic() >= self.deadline:
                raise SearchDeadline()
            if depth == 0 or self.terminal(node):
                return self.node_value(node)
            idx = actors[ply]
            next_ply = (ply + 1) % len(actors)
            next_depth = depth - int(next_ply == 0)
            if idx != self.index and (idx not in active or self.search_position(node, idx) is None):
                if self.search_position(node, idx) is None:
                    # Hidden non-sampled turns consume time; never recover a private configuration.
                    future = node.deep_copy()
                    future.data.timeleft -= 1
                    future.get_agent_state(idx).scared_timer = max(0, future.get_agent_state(idx).scared_timer - 1)
                else:
                    future = node.generate_successor(idx, self.passive_action(node, idx))
                return visit(future, next_ply, next_depth, alpha, beta)
            options = [(a, node.generate_successor(idx, a)) for a in node.get_legal_actions(idx)]
            if idx == self.index:
                options.sort(key=lambda pair: self.node_value(pair[1]), reverse=True)
                value = -float('inf')
                for action, future in options:
                    value = max(value, visit(future, next_ply, next_depth, alpha, beta))
                    alpha = max(alpha, value)
                    if not stochastic and alpha >= beta:
                        break
                return value
            if stochastic:
                probabilities = dict(self.opponent_actions(node, idx))
                return sum(probabilities[a] * visit(future, next_ply, next_depth, -float('inf'), float('inf'))
                           for a, future in options)
            options.sort(key=lambda pair: self.node_value(pair[1]))
            value = float('inf')
            for action, future in options:
                value = min(value, visit(future, next_ply, next_depth, alpha, beta))
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value

        for depth in range(1, 4):
            complete = {}
            try:
                for action in sorted(values, key=values.get, reverse=True):
                    complete[action] = visit(state.generate_successor(self.index, action), 1, depth,
                                             -float('inf'), float('inf'))
            except SearchDeadline:
                break
            values = complete
            best_action = max(values, key=lambda a: (values[a], a != Directions.STOP, a))
            self.last_search_depth, self.last_search_values = depth, values
        return best_action

    def sample_world(self, state):
        world = state.deep_copy()
        for i, belief in self.shared.beliefs.items():
            if self.search_position(state, i) is not None:
                continue
            enemy = state.get_agent_state(i)
            def compatible(p):
                own_side = (p[0] < self.shared.width // 2) == state.is_on_red_team(i)
                teammates = [self.search_position(state, j) for j in self.get_team(state)]
                unseen = all(q is None or abs(p[0] - q[0]) + abs(p[1] - q[1]) > 5 for q in teammates)
                readings = state.get_agent_distances()
                observer = self.search_position(state, self.index)
                compatible_noise = not readings or state.get_distance_prob(
                    abs(p[0] - observer[0]) + abs(p[1] - observer[1]), readings[i]) > 0
                return p in self.shared.cells and own_side != enemy.is_pacman and unseen and compatible_noise
            support = [(p, w) for p, w in belief.items() if compatible(p) and w > 0]
            if not support:
                support = [(p, 1.0) for p in self.shared.cells if compatible(p)]
            if support:
                positions, weights = zip(*support)
                position = self.rng.choices(positions, weights=weights, k=1)[0]
                world.data.agent_states[i].configuration = Configuration(position, Directions.STOP)
        return world

    def simulated_step(self, state, action):
        node = state.generate_successor(self.index, action)
        opponents = self.get_opponents(state)
        for i in self.round_actors(state)[1:]:
            if self.terminal(node):
                break
            if self.search_position(node, i) is None:
                node = node.deep_copy()
                node.data.timeleft -= 1
                node.get_agent_state(i).scared_timer = max(0, node.get_agent_state(i).scared_timer - 1)
                continue
            if i in opponents:
                options = self.opponent_actions(node, i)
                choice = self.rng.choices([a for a, p in options], weights=[p for a, p in options], k=1)[0]
            else:
                choice = self.passive_action(node, i)
            node = node.generate_successor(i, choice)
        return node

    def mcts(self, state, beliefs=False):
        # Open-loop UCT; independent role-compatible determinizations in belief mode.
        root = dict(n=0, total=0.0, children={})
        initial = self.action_values(state)
        fallback = max(initial, key=lambda a: (initial[a], a != Directions.STOP, a))
        iterations = 0
        base = self.node_value(state)
        while time.monotonic() < self.deadline:
            world = self.sample_world(state) if beliefs else state.deep_copy()
            node, path = root, [root]
            try:
                for depth in range(6):
                    if time.monotonic() >= self.deadline:
                        raise SearchDeadline()
                    if self.terminal(world):
                        break
                    legal = world.get_legal_actions(self.index)
                    priors = self.action_values(world)
                    top = max(priors.values())
                    raw = {a: math.exp(max(-12, (v - top) / 10.0)) for a, v in priors.items()}
                    mass = sum(raw.values())
                    priors = {a: w / mass for a, w in raw.items()}
                    untried = [a for a in legal if a not in node['children']]
                    expanded = bool(untried)
                    if expanded:
                        action = max(untried, key=lambda a: (priors[a], a != Directions.STOP, a))
                        node['children'][action] = dict(n=0, total=0.0, children={})
                    else:
                        action = max(legal, key=lambda a: node['children'][a]['total'] / max(1, node['children'][a]['n']) +
                                     1.2 * math.sqrt(math.log(node['n'] + 1) / max(1, node['children'][a]['n'])) +
                                     0.5 * priors[a] / (1 + node['children'][a]['n']))
                    world = self.simulated_step(world, action)
                    node = node['children'][action]
                    path.append(node)
                    if expanded:
                        for _ in range(3):
                            if time.monotonic() >= self.deadline:
                                raise SearchDeadline()
                            if self.terminal(world):
                                break
                            rollout = self.action_values(world)
                            action = self.rng.choice(list(rollout)) if self.rng.random() < 0.15 else max(rollout, key=rollout.get)
                            world = self.simulated_step(world, action)
                        break
                reward = math.tanh((self.node_value(world) - base) / 35.0)
            except SearchDeadline:
                break
            for entry in path:
                entry['n'] += 1
                entry['total'] += reward
            iterations += 1
        self.last_iterations = iterations
        self.last_mcts_values = {a: c['total'] / c['n'] for a, c in root['children'].items() if c['n']}
        visited = [a for a in root['children'] if root['children'][a]['n']]
        if not visited:
            return fallback
        return max(visited, key=lambda a: (root['children'][a]['n'], self.last_mcts_values[a]))

    def mdp(self, state):
        start = nearest_point(self.search_position(state, self.index))
        goal = nearest_point(self.projected_goal(state))
        mine = state.get_agent_state(self.index)
        frontier, positions = collections.deque([(start, 0)]), {start}
        while frontier and len(positions) < 120 and time.monotonic() < self.deadline:
            pos, depth = frontier.popleft()
            if depth >= 12:
                continue
            for q in self.shared.adj[pos]:
                if q not in positions:
                    positions.add(q)
                    frontier.append((q, depth + 1))
        # Absorbing subtask rewards avoid repeatedly rewarding a deposit or food pickup.
        terminal = {'collect': 10.0, 'capsule': 18.0, 'return': 22.0 + 6 * mine.num_carrying,
                    'defend': 24.0, 'escape': 18.0, 'guard': 12.0}.get(self.role, 10.0)
        foods = set(self.get_food(state).as_list())
        capsules = set(self.get_capsules(state))
        risks = {p: min(0.95, self.risk_at(p, state)) for p in positions}
        if goal in capsules:
            risks[goal] = 0.0
        values = {p: terminal if p == goal else -0.8 * self.distance(p, goal) for p in positions}
        discount = 0.88

        def bellman(p, q):
            hazard = risks[q] if q in risks else min(0.95, self.risk_at(q, state))
            future = values.get(q, -0.8 * self.distance(q, goal))
            reward = -1.0 + (0.4 if self.role == 'collect' and q in foods else 0.0)
            if self.role in ('defend', 'guard') and self.on_enemy_side(q):
                reward -= 8.0
            death = -20.0 - 8 * mine.num_carrying
            return reward + discount * ((1 - hazard) * (0.9 * future + 0.1 * values[p]) + hazard * death)

        self.last_mdp_iterations, self.last_mdp_residual = 0, float('inf')
        for iteration in range(100):
            new = {}
            for p in positions:
                if time.monotonic() >= self.deadline:
                    break
                new[p] = terminal if p == goal else max(bellman(p, q) for q in self.shared.adj[p] + [p])
            if len(new) != len(positions):
                break
            residual = max(abs(new[p] - values[p]) for p in positions)
            values = new
            self.last_mdp_iterations, self.last_mdp_residual = iteration + 1, residual
            if residual < 0.001:
                break
        choices = {}
        for action in state.get_legal_actions(self.index):
            future = state.generate_successor(self.index, action)
            q = nearest_point(self.search_position(future, self.index))
            b = future.get_agent_state(self.index)
            deposited = b.num_returned - mine.num_returned
            died = q == self.start and self.distance(start, self.start) > 2
            projected_risk = self.risk_at(q, future)
            choices[action] = values.get(q, -0.8 * self.distance(q, goal)) + 20 * deposited
            choices[action] -= 25 * projected_risk + (35 + 8 * mine.num_carrying if died else 0)
            if action == Directions.STOP and not (self.role == 'guard' and q == goal):
                choices[action] -= 1.0
        self.last_mdp_values = choices
        return max(choices, key=lambda a: (choices[a], a != Directions.STOP, a))

    def hierarchical(self, state):
        pos = nearest_point(self.search_position(state, self.index))
        target = self.projected_goal(state)
        macro = self.role
        # A loaded visible invader's closest exit is more useful than chasing its old square.
        if self.role == 'defend' and state.get_agent_state(self.index).scared_timer <= 0:
            invaders = [state.get_agent_state(i) for i in self.get_opponents(state)
                        if state.get_agent_state(i).is_pacman and self.search_position(state, i) is not None]
            enemy = self.assigned_invader(state)
            if enemy is not None:
                exit_point = min(self.shared.border, key=lambda p: self.distance(enemy.get_position(), p))
                if enemy.num_carrying >= 3 and self.distance(pos, exit_point) <= self.distance(enemy.get_position(), exit_point) + 1:
                    target, macro = exit_point, 'intercept'
        objective = target
        if self.on_enemy_side(pos) != self.on_enemy_side(target):
            gateways = [p for p in self.shared.border if
                        (p[0] + (1 if self.red else -1), p[1]) in self.shared.cells]
            def gateway_cost(p):
                other = (p[0] + (1 if self.red else -1), p[1])
                return self.distance(pos, p) + self.distance(p, objective) + 15 * max(
                    self.risk_at(p, state), self.risk_at(other, state))
            target = min(gateways, key=gateway_cost)
            if pos == target and self.on_enemy_side(objective):
                target = (pos[0] + (1 if self.red else -1), pos[1])
            if self.role not in ('return', 'escape'):
                macro = 'border'
        elif self.distance(pos, target) > 8:
            junctions = [p for p in self.shared.cells if len(self.shared.adj[p]) >= 3 and
                         self.distance(pos, p) + self.distance(p, objective) <= self.distance(pos, objective) + 3 and
                         0 < self.distance(pos, p) <= 6]
            if junctions:
                target = min(junctions, key=lambda p: self.distance(p, objective) + 15 * self.risk_at(p, state) +
                             self.shared.depth.get(p, 0) * self.risk_at(p, state))
                macro = 'waypoint'
        self.last_macro = (macro, target)
        if macro == 'intercept' and pos == target and Directions.STOP in state.get_legal_actions(self.index):
            captures = [a for a in state.get_legal_actions(self.index) if any(
                self.distance(state.generate_successor(self.index, a).get_agent_position(self.index), e.get_position()) <= 0.7
                for e in invaders)]
            if not captures:
                return Directions.STOP
        return self.astar(state, target)

    def neural_value(self, features):
        model = self.model
        hidden = [math.tanh(sum(x * w for x, w in zip(features, column)) + bias)
                  for column, bias in zip(model['w1'], model['b1'])]
        return sum(h * w for h, w in zip(hidden, model['w2'])) + model['b2']

    def decide(self, state):
        if self.mode == 'minimax':
            return self.adversarial(state)
        if self.mode == 'expectimax':
            return self.adversarial(state, stochastic=True)
        if self.mode in ('uct', 'belief_mcts'):
            return self.mcts(state, beliefs=self.mode == 'belief_mcts')
        if self.mode == 'mdp':
            return self.mdp(state)
        if self.mode == 'hierarchical':
            return self.hierarchical(state)
        if self.mode in ('qlearning', 'evolution'):
            return self.greedy(state)
        if self.mode == 'neural':
            return max(state.get_legal_actions(self.index), key=lambda a: self.neural_value(self.features(state, a)))
        if self.mode in ('adaptive', 'hybrid'):
            policy = self.select_policy(state, hybrid=self.mode == 'hybrid')
            if policy == 'hierarchical':
                return self.hierarchical(state)
            if policy == 'belief_mcts':
                return self.mcts(state, beliefs=True)
            return self.adversarial(state, stochastic=policy == 'expectimax')
        return self.astar(state)

    def select_policy(self, state, hybrid=False):
        pos = self.search_position(state, self.index)
        nearby = [i for i in self.get_opponents(state) if self.search_position(state, i) is not None and
                  self.distance(pos, self.search_position(state, i)) <= 5]
        if hybrid:
            if nearby:
                policy = 'minimax' if self.risk_at(pos, state) > 0.2 or self.role in ('defend', 'escape', 'return') else 'expectimax'
            elif self.risk_at(pos, state) > 0.15:
                policy = 'belief_mcts'
            else:
                policy = 'hierarchical'
        elif not nearby:
            policy = 'hierarchical'
        else:
            evidence = [self.shared.behaviour[i] for i in nearby]
            count = sum(v['moves'] for v in evidence)
            rate = sum(v['chases'] for v in evidence) / max(1, count)
            # Wilson interval avoids treating a handful of chases as certain behaviour.
            z = 1.96
            centre = (rate + z * z / (2 * max(1, count))) / (1 + z * z / max(1, count))
            radius = z * math.sqrt(rate * (1 - rate) / max(1, count) + z * z / (4 * max(1, count) ** 2)) / (1 + z * z / max(1, count))
            previous = getattr(self, 'policy', 'expectimax')
            policy = 'minimax' if count >= 12 and (centre - radius > 0.55 or
                     previous == 'minimax' and centre + radius >= 0.45) else 'expectimax'
            self.policy_confidence = (max(0.0, centre - radius), min(1.0, centre + radius), count)
        self.policy = policy
        return policy

"""Preserve complete adversarial values; prefer earlier progress on exact root ties."""
import hashlib
import json
from pathlib import Path
import re
from contest.game import Directions


class ProgressAdversarialAgent(StrategyAgent):
    def action_values(self, state):
        values = super().action_values(state)
        if state is getattr(self, '_tie_root_state', None):
            self._tie_initial_values = values
        return values

    def adversarial(self, state, stochastic=False):
        self._tie_root_state = state
        self._tie_initial_values = None
        self.last_adversarial_tie_broken = False
        try:
            original = super().adversarial(state,stochastic)
        finally:
            self._tie_root_state = None
        if self.last_search_depth == 0:
            return original
        values = self.last_search_values
        initial = self._tie_initial_values
        action = max(values, key=lambda a: (values[a], initial[a], a != Directions.STOP, a))
        self.last_adversarial_tie_broken = action != original
        return action



"""Versioned opponent model that considers both own-team invaders."""
import hashlib
import json
import math
from pathlib import Path

from contest.game import Directions


class TeamTargetOpponentMixin:
    def opponent_actions(self, state, index):
        opponent = state.get_agent_state(index)
        invaders = [state.get_agent_state(i).get_position() for i in self.get_team(state)
                    if state.get_agent_state(i).is_pacman and state.get_agent_state(i).get_position() is not None]
        if opponent.is_pacman or not invaders:
            return super().opponent_actions(state, index)
        actions = state.get_legal_actions(index)
        values = []
        for action in actions:
            future = state.generate_successor(index, action)
            pos = future.get_agent_state(index).get_position()
            distance = min(self.distance(pos, target) for target in invaders)
            value = (1 if opponent.scared_timer > 1 else -1) * distance
            value -= .2 * int(action == Directions.STOP)
            if pos == state.get_initial_agent_position(index) and self.distance(opponent.get_position(), pos) > 2:
                value = -20 - 5 * opponent.num_carrying
            values.append(value)
        maximum = max(values)
        weights = [math.exp(max(-12, value - maximum)) for value in values]
        total = sum(weights)
        evidence = self.shared.behaviour.get(index, {}).get('moves', 0)
        uniform = .1 + .3 / (1 + evidence / 8.)
        return [(action, (1-uniform)*weight/total + uniform/len(actions))
                for action, weight in zip(actions, weights)]



class TeamTargetAgent(TeamTargetOpponentMixin, ProgressAdversarialAgent):
    pass

"""Loaded-turn safety with possible intervening teammate capsule/finish rescues."""
import hashlib
import json
from pathlib import Path
import time

from contest.game import Directions


class LoadedTurnDeadline(Exception):
    pass


class LoadedTurnGuardMixin:
    def reply_safe_action(self, state, proposed, deadline):
        self.last_reply_guard_changed = False
        self.last_reply_guard_complete = True
        mine = state.get_agent_state(self.index)
        if not mine.is_pacman or mine.num_carrying <= 0:
            return proposed
        futures = {}

        def check():
            if time.monotonic() >= deadline:
                raise LoadedTurnDeadline()

        def dangerous(action):
            check()
            future = state.generate_successor(self.index, action)
            futures[action] = future
            after = future.get_agent_state(self.index)
            if after.num_carrying < mine.num_carrying and after.num_returned == mine.num_returned:
                return True
            if not after.is_pacman or future.is_over() or future.data.timeleft <= 0:
                return False
            for offset in (1, 3):
                if future.data.timeleft < offset:
                    continue
                enemy_index = (self.index+offset) % state.get_num_agents()
                if enemy_index not in self.get_opponents(state):
                    continue
                enemy = future.get_agent_state(enemy_index)
                position = enemy.get_position()
                if position is None or enemy.scared_timer > 0:
                    continue
                own_position = after.get_position()
                if abs(position[0]-own_position[0])+abs(position[1]-own_position[1]) > 2:
                    continue
                capturable = False
                for response in future.get_legal_actions(enemy_index):
                    check()
                    outcome = future.generate_successor(enemy_index, response).get_agent_state(self.index)
                    if outcome.num_carrying < after.num_carrying and outcome.num_returned == after.num_returned:
                        capturable = True
                        break
                if not capturable:
                    continue
                if offset == 3:
                    teammate = (self.index+2) % state.get_num_agents()
                    rescue = False
                    for move in future.get_legal_actions(teammate):
                        check()
                        intervening = future.generate_successor(teammate, move)
                        if (intervening.get_agent_state(enemy_index).scared_timer > 0 or
                                (intervening.is_over() and self.get_score(intervening) > 0)):
                            rescue = True
                            break
                    if rescue:
                        continue
                return True
            return False

        try:
            if not dangerous(proposed):
                return proposed
            safe = [a for a in state.get_legal_actions(self.index) if a != proposed and not dangerous(a)]
            if not safe:
                return proposed
            def value(action):
                check()
                score = self.neural_value(self.features(state, action)) if self.mode == 'neural' else self.node_value(futures[action])
                return (score, action != Directions.STOP, action)
            selected = max(safe, key=value)
            check()
        except LoadedTurnDeadline:
            self.last_reply_guard_complete = False
            return proposed
        self.last_reply_guard_changed = selected != proposed
        return selected

    def choose_action(self, state):
        started = time.monotonic()
        proposed = super().choose_action(state)
        return self.reply_safe_action(state, proposed, min(started+.7, time.monotonic()+.01))



class FullTurnGuardAgent(LoadedTurnGuardMixin, TeamTargetAgent):
    pass


def create_team(first_index, second_index, is_red, inference='exact', budget=0.07, num_training=0):
    if inference not in ('exact','particle'):
        raise ValueError('Unknown inference mode')
    shared = SharedTeam(inference)
    agents=[FullTurnGuardAgent(first_index,shared,0,budget),FullTurnGuardAgent(second_index,shared,1,budget)]
    for agent in agents:agent.mode='expectimax'
    return agents


class FullObservationMixin:
    def observation_function(self, game_state):
        # Some external runners may already supply a sanitized observation.
        if any(game_state.get_agent_state(i).configuration is None
               for i in range(game_state.get_num_agents())):
            return game_state.deep_copy()
        observation = super().observation_function(game_state)
        for index in self.get_opponents(observation):
            observed = observation.get_agent_state(index)
            if observed.configuration is None:
                actual = game_state.get_agent_state(index).configuration
                observed.configuration = Configuration(actual.get_position(), actual.get_direction())
        return observation


class FullStateFullTurnGuardAgent(FullObservationMixin, FullTurnGuardAgent):
    pass


def create_team(first_index, second_index, is_red, inference='exact', budget=0.2, num_training=0):
    if inference not in ('exact','particle'):
        raise ValueError('Unknown inference mode')
    shared = SharedTeam(inference)
    agents=[FullStateFullTurnGuardAgent(first_index,shared,0,budget),FullStateFullTurnGuardAgent(second_index,shared,1,budget)]
    for agent in agents:agent.mode='expectimax'
    return agents


"""Scalar world snapshots and graph facts shared by distinct team policies."""
from collections import deque
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import FrozenSet, Optional, Tuple


@dataclass(frozen=True)
class TLActor:
    index: int
    position: Optional[Tuple[float, float]]
    pacman: bool
    carrying: int
    returned: int
    scared: int


@dataclass(frozen=True)
class TLWorld:
    actors: Tuple[TLActor, ...]
    team: Tuple[int, ...]
    enemies: Tuple[int, ...]
    food: FrozenSet[Tuple[int, int]]
    defended_food: FrozenSet[Tuple[int, int]]
    capsules: FrozenSet[Tuple[int, int]]
    defended_capsules: FrozenSet[Tuple[int, int]]
    spawns: Tuple[Tuple[int, int], ...]
    score: int
    timeleft: int
    next_actor: int
    width: int
    height: int
    red: bool
    win_threshold: int
    own_returned: int
    enemy_returned: int
    terminal: bool

    def actor(self, index):
        return self.actors[index]

    def enemy_side(self, point):
        return (point[0] < self.width // 2) != self.red

    @classmethod
    def from_agent(cls, agent, state):
        actors = []
        for index in range(state.get_num_agents()):
            source = state.get_agent_state(index)
            position = source.get_position()
            actors.append(TLActor(index, tuple(position) if position is not None else None,
                                  bool(source.is_pacman), source.num_carrying,
                                  source.num_returned, source.scared_timer))
        team = tuple(sorted(agent.get_team(state)))
        enemies = tuple(sorted(agent.get_opponents(state)))
        own_returned = sum(actors[i].returned for i in team)
        enemy_returned = sum(actors[i].returned for i in enemies)
        threshold = max(0, math.ceil(state.data.layout.total_food / 2 - 2))
        terminal = (state.is_over() or state.data.timeleft <= 0 or
                    own_returned >= threshold or enemy_returned >= threshold)
        moved = state.data._agent_moved
        next_actor = (moved + 1) % len(actors) if moved is not None else agent.index
        return cls(tuple(actors), team, enemies,
                   frozenset(agent.get_food(state).as_list()),
                   frozenset(agent.get_food_you_are_defending(state).as_list()),
                   frozenset(agent.get_capsules(state)),
                   frozenset(agent.get_capsules_you_are_defending(state)),
                   tuple(tuple(state.get_initial_agent_position(i)) for i in range(len(actors))),
                   agent.get_score(state), state.data.timeleft, next_actor,
                   state.data.layout.width, state.data.layout.height, bool(agent.red),
                   threshold, own_returned, enemy_returned, bool(terminal))


def tl_arrival_plies(actor, next_actor, turns, count=4):
    if count < 1 or not 0 <= actor < count or not 0 <= next_actor < count or turns < 0:
        raise ValueError('Invalid actor arrival')
    if turns == 0:
        return 0
    return (actor - next_actor) % count + 1 + (int(math.ceil(turns)) - 1) * count


class TLGeometry:
    def __init__(self, adj, border, enemy_border):
        self.adj = MappingProxyType({p: tuple(sorted(neighbours)) for p, neighbours in adj.items()})
        self.border = tuple(sorted(border))
        self.enemy_border = tuple(sorted(enemy_border))
        self._distances = {}
        self._pockets = {}
        protected = set(self.border) | set(self.enemy_border)
        degree = {p: len(neighbours) for p, neighbours in self.adj.items()}
        queue = deque(sorted(p for p, n in degree.items() if n <= 1 and p not in protected))
        removed = set()
        while queue:
            p = queue.popleft()
            if p in removed:
                continue
            removed.add(p)
            for q in self.adj[p]:
                degree[q] -= 1
                if degree[q] == 1 and q not in protected and q not in removed:
                    queue.append(q)
        remaining = set(removed)
        while remaining:
            seed = min(remaining)
            component, pending, exits = {seed}, [seed], set()
            remaining.remove(seed)
            while pending:
                p = pending.pop()
                for q in self.adj[p]:
                    if q not in removed:
                        exits.add(q)
                    elif q in remaining:
                        remaining.remove(q)
                        component.add(q)
                        pending.append(q)
            if len(exits) == 1:
                record = frozenset(component), next(iter(exits))
                self._pockets.update((p, record) for p in component)

    @classmethod
    def from_agent(cls, agent):
        delta = 1 if agent.red else -1
        enemy_border = [(x + delta, y) for x, y in agent.shared.border
                        if (x + delta, y) in agent.shared.adj]
        return cls(agent.shared.adj, agent.shared.border, enemy_border)

    def pocket(self, point):
        return self._pockets.get(point)

    def _anchors(self, point):
        if point is None:
            return ()
        x, y = point
        result = []
        for a in {math.floor(x), math.ceil(x)}:
            for b in {math.floor(y), math.ceil(y)}:
                cell = a, b
                if cell in self.adj:
                    result.append((cell, abs(x - a) + abs(y - b)))
        return tuple(sorted(result))

    def distances(self, origin):
        if origin not in self.adj:
            return {}
        if origin not in self._distances:
            distances, queue = {origin: 0}, deque([origin])
            while queue:
                p = queue.popleft()
                for q in self.adj[p]:
                    if q not in distances:
                        distances[q] = distances[p] + 1
                        queue.append(q)
            self._distances[origin] = distances
        return self._distances[origin]

    def distance(self, a, b):
        if a is None or b is None:
            return float('inf')
        if a == b:
            return 0.0
        choices = [da + self.distances(pa).get(pb, float('inf')) + db
                   for pa, da in self._anchors(a) for pb, db in self._anchors(b)]
        return min(choices) if choices else float('inf')

    def path(self, origin, target):
        if origin not in self.adj or target not in self.adj:
            return None
        distance = self.distances(target)
        if origin not in distance:
            return None
        path = [origin]
        while path[-1] != target:
            p = path[-1]
            path.append(min(q for q in self.adj[p] if distance.get(q) == distance[p] - 1))
        return tuple(path)


"""Bounded team tasks, complete-excursion estimates and explicit execution state."""
from dataclasses import dataclass
import heapq
import math
import time
from typing import Optional, Tuple



@dataclass(frozen=True)
class TLTask:
    actor: int
    kind: str
    target: Tuple[int, int]
    utility: float = 0.0
    duration: float = 0.0
    exit: Optional[Tuple[int, int]] = None
    load_limit: int = 0
    enemy: Optional[int] = None
    beneficiary: Optional[int] = None
    deadline: Optional[int] = None
    risk: float = 0.0
    reward: float = 0.0
    route: Tuple[Tuple[int, int], ...] = ()


@dataclass(frozen=True)
class TLRoute:
    exit: Optional[Tuple[int, int]]
    steps: float
    feasible: bool
    reason: str
    risk: float = 0.0
    margin: float = float('inf')


@dataclass(frozen=True)
class TLJointPlan:
    tasks: Tuple[TLTask, ...]
    utility: float

    def for_actor(self, index):
        return next(task for task in self.tasks if task.actor == index)


class TLTaskDeadline(Exception):
    pass


def tl_check_task_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TLTaskDeadline()


def tl_capture_deadline(world, geometry, victim, point):
    result = float('inf')
    own = world.actor(victim)
    for index in world.enemies:
        enemy = world.actor(index)
        if enemy.position is None:
            continue
        if not world.enemy_side(point) and own.scared <= 0:
            continue
        distance = geometry.distance(enemy.position, point)
        if not math.isfinite(distance):
            continue
        # This is a conservative arrival bound, not a simulated opponent policy.
        turns = max(math.ceil(distance), enemy.scared if world.enemy_side(point) else 0)
        result = min(result, tl_arrival_plies(index, world.next_actor, turns, len(world.actors)))
    return result


def tl_bank_route(world, geometry, index, target, hazards=None, deadline=float('inf')):
    origin = world.actor(index).position
    anchors = geometry._anchors(origin)
    if not anchors:
        return None
    start = min(anchors, key=lambda item: (item[1], item[0]))[0]
    to_target = geometry.distances(target)
    direct = to_target.get(start, float('inf'))
    if not math.isfinite(direct):
        return None
    count = len(world.actors)
    offset = (index - world.next_actor) % count
    available = max(0, (world.timeleft - offset + count - 1) // count)
    limit = min(int(direct) + 6, available)
    queue = [(direct, 0.0, start, 0, 0.0, (start,))]
    best = {(start, 0): 0.0}
    while queue:
        tl_check_task_deadline(deadline)
        _, cost, point, steps, danger, path = heapq.heappop(queue)
        if cost != best.get((point, steps)):
            continue
        if point == target:
            return path, danger
        if steps >= limit:
            continue
        for future in geometry.adj[point]:
            arrival = tl_arrival_plies(index, world.next_actor, steps + 1, len(world.actors))
            threat_time = (hazards[future] if hazards is not None else
                           tl_capture_deadline(world, geometry, index, future))
            margin = threat_time - arrival
            risk = 1.0 if margin <= 0 else max(0.0, (4 - margin) / 4)
            value = cost + 1 + 20 * risk
            key = future, steps + 1
            if value < best.get(key, float('inf')):
                best[key] = value
                heapq.heappush(queue, (value + to_target.get(future, float('inf')), value,
                                      future, steps + 1, max(danger, risk), path + (future,)))
    return None


def tl_excursion(world, geometry, index, target):
    mine = world.actor(index)
    if mine.position is None or target not in geometry.adj:
        return TLRoute(None, float('inf'), False, 'unreachable')
    exits = [(geometry.distance(target, p), p) for p in geometry.border]
    home_steps, exit_cell = min(exits, default=(float('inf'), None))
    outward = geometry.distance(mine.position, target)
    steps = outward + home_steps
    if not math.isfinite(steps):
        return TLRoute(exit_cell, steps, False, 'unreachable')
    if tl_arrival_plies(index, world.next_actor, steps, len(world.actors)) > world.timeleft:
        return TLRoute(exit_cell, steps, False, 'clock')
    pocket = geometry.pocket(target)
    margin = float('inf')
    if pocket is not None and world.enemy_side(target):
        component, mouth = pocket
        leave_steps = outward + geometry.distance(target, mouth)
        own_arrival = tl_arrival_plies(index, world.next_actor, leave_steps, len(world.actors))
        enemy_arrival = tl_capture_deadline(world, geometry, index, mouth)
        margin = enemy_arrival - own_arrival
        own_to_mouth = geometry.distance(mine.position, mouth)
        close_threat = any(enemy.position is not None and
            geometry.distance(enemy.position, mouth) <= 6 and enemy.scared <= 3
            for enemy in (world.actor(i) for i in world.enemies))
        # A distant worst-case arrival is not a commitment made now. Revalidate
        # near the mouth instead of declaring every opening harvest impossible.
        if margin <= 0 and own_to_mouth <= 6 and close_threat:
            return TLRoute(exit_cell, steps, False, 'blocked-pocket-exit', 1.0, margin)
    endpoint_deadline = tl_capture_deadline(world, geometry, index, target)
    endpoint_arrival = tl_arrival_plies(index, world.next_actor, outward, len(world.actors))
    risk = max(0.0, min(1.0, (8 - (endpoint_deadline - endpoint_arrival)) / 12))
    return TLRoute(exit_cell, steps, True, 'reachable', risk, margin)


def tl_candidates(world, geometry, index, risk=None, deadline=float('inf')):
    try:
        tl_check_task_deadline(deadline)
        return tl_candidate_values(world, geometry, index, risk, deadline)
    except TLTaskDeadline:
        return ()


def tl_candidate_values(world, geometry, index, risk, deadline):
    mine = world.actor(index)
    position = mine.position
    if position is None:
        return ()
    # Half steps remain exact in the world; task targets are legal integer cells.
    anchors = geometry._anchors(position)
    if not anchors:
        return ()
    cell = min(anchors, key=lambda x: (x[1], x[0]))[0]
    options = []
    borders = sorted(geometry.border, key=lambda p: (geometry.distance(position, p), p))[:2]
    enemy_need = max(0, world.win_threshold - world.enemy_returned)
    own_need = max(0, world.win_threshold - world.own_returned)
    if mine.carrying > 0:
        banks = []
        hazards = {}
        for point in geometry.adj:
            tl_check_task_deadline(deadline)
            hazards[point] = tl_capture_deadline(world, geometry, index, point)
        # Distinct near, best-margin and distant exits keep the search bounded.
        possible = set(borders)
        if geometry.border:
            possible.add(max(geometry.border, key=lambda p: (geometry.distance(position, p), p)))
            delta = 1 if world.red else -1
            possible.add(max(geometry.border, key=lambda p: (
                hazards.get((p[0] + delta, p[1]), float('inf')) -
                tl_arrival_plies(index, world.next_actor, geometry.distance(position, p)), p)))
        for target in sorted(possible):
            planned = tl_bank_route(world, geometry, index, target, hazards, deadline)
            if planned is None:
                continue
            path, danger = planned
            distance = len(path) - 1
            expected = mine.carrying
            utility = 22 * expected / (distance + 3)
            utility += 250 if expected >= own_need > 0 else 0
            utility += 20 if tl_arrival_plies(index, world.next_actor, distance + 3) >= world.timeleft else 0
            utility -= (10 + 8 * mine.carrying) * danger / (distance + 3)
            banks.append(TLTask(index, 'bank', target, utility, distance, target,
                                risk=danger, reward=float(expected), route=path))
        options.extend(sorted(banks, key=lambda task: (-task.utility, task.target))[:2])

    # Rank a bounded set before doing route/pocket checks. Dense food is valuable
    # only when the complete excursion still permits a deposit.
    foods = sorted(world.food, key=lambda p: (geometry.distance(position, p), p))[:12]
    harvests = []
    for target in foods:
        tl_check_task_deadline(deadline)
        route = tl_excursion(world, geometry, index, target)
        if not route.feasible:
            continue
        proximity = sum(geometry.distance(target, p) <= 2 for p in world.food)
        gain = min(3, proximity, max(1, own_need - mine.carrying))
        danger = route.risk
        if risk is not None:
            danger = max(danger, min(1.0, max(0.0, float(risk(target, index)))))
        utility = 22 * (mine.carrying + gain) / (route.steps + 4)
        utility -= (2 + .8 * mine.carrying) * danger / (route.steps + 4)
        harvests.append(TLTask(index, 'harvest', target, utility, route.steps, route.exit,
                               mine.carrying + gain, risk=danger, reward=float(gain)))
    options.extend(sorted(harvests, key=lambda t: (-t.utility, t.target))[:3])

    if not mine.pacman and mine.scared == 0:
        for enemy_index in world.enemies:
            enemy = world.actor(enemy_index)
            if not enemy.pacman or enemy.position is None:
                continue
            target = min(geometry.adj, key=lambda p: (geometry.distance(p, enemy.position), p))
            if world.enemy_side(target):
                continue
            distance = geometry.distance(position, target)
            reward = enemy.carrying + 1.5
            emergency = 100 if enemy.carrying >= enemy_need > 0 else 0
            options.append(TLTask(index, 'intercept', target, 24 * reward / (distance + 3) + emergency,
                                  distance, enemy=enemy_index, reward=reward))

    for capsule in sorted(world.capsules):
        tl_check_task_deadline(deadline)
        distance = geometry.distance(position, capsule)
        if not math.isfinite(distance) or distance > 12:
            continue
        arrival = tl_arrival_plies(index, world.next_actor, distance, len(world.actors))
        for beneficiary in world.team:
            carrier = world.actor(beneficiary)
            if not carrier.pacman or carrier.position is None or carrier.carrying <= 0:
                continue
            capture_plies = tl_capture_deadline(world, geometry, beneficiary, carrier.position)
            if not math.isfinite(capture_plies) or capture_plies >= 40 or arrival >= capture_plies:
                continue
            active = any(world.actor(i).position is not None and world.actor(i).scared <= distance + 2
                         for i in world.enemies)
            if not active:
                continue
            utility = 20 * carrier.carrying / (distance + 3)
            options.append(TLTask(index, 'capsule', capsule, utility, distance,
                                  beneficiary=beneficiary, deadline=world.timeleft - int(capture_plies),
                                  reward=float(carrier.carrying)))

    if not mine.pacman:
        if world.defended_food:
            target = min(geometry.border, key=lambda p: (
                sum(geometry.distance(p, q) for q in world.defended_food) / len(world.defended_food) +
                .3 * geometry.distance(position, p), p), default=cell)
        else:
            target = min(borders, key=lambda p: geometry.distance(position, p), default=cell)
        hidden_cargo = sum(world.actor(i).carrying for i in world.enemies
                           if world.actor(i).pacman and world.actor(i).position is None)
        utility = (.6 + hidden_cargo) / (geometry.distance(position, target) + 2)
        if world.score > 0 and world.timeleft < 160:
            utility += 2
        options.append(TLTask(index, 'guard', target, utility, geometry.distance(position, target)))
    elif not harvests and not mine.carrying:
        target = min(borders, key=lambda p: geometry.distance(position, p), default=cell)
        options.append(TLTask(index, 'escape', target, .1, geometry.distance(position, target), target))
    options.append(TLTask(index, 'wait', cell, -3.0))
    return tuple(sorted(options, key=lambda t: (-t.utility, t.kind, t.target))[:8])


def tl_joint_plans(tasks_by_actor, world):
    indices = sorted(tasks_by_actor)
    if len(indices) != 2:
        raise ValueError('A team plan requires two agents')
    first, second = indices
    result = []
    for a in tasks_by_actor[first]:
        for b in tasks_by_actor[second]:
            utility = a.utility + b.utility
            if a.kind == b.kind == 'harvest' and a.target == b.target:
                utility -= 8 + max(a.utility, b.utility)
            if a.kind == b.kind == 'intercept' and a.enemy == b.enemy and a.target == b.target:
                utility -= min(a.utility, b.utility) + 2
            if a.kind == b.kind == 'capsule' and a.target == b.target:
                utility -= max(a.utility, b.utility) + 2
            if a.kind == b.kind == 'guard' and a.target == b.target:
                utility -= 2
            for helper, carrier in [(a, b), (b, a)]:
                if helper.kind == 'capsule' and helper.beneficiary == carrier.actor and carrier.kind in ('bank', 'escape'):
                    utility -= helper.utility * (1 - max(0, min(1, carrier.risk)))
            result.append(TLJointPlan((a, b), utility))
    return tuple(result)


class TLExecution:
    def __init__(self, task, world):
        self.task = task
        self.started_at = world.timeleft
        self.origin = world.actor(task.actor)
        self.enemy_timers = {i: world.actor(i).scared for i in world.enemies}
        self.previous = self.origin
        self.last_time = world.timeleft
        self.stagnant = 0
        self.status, self.reason, self.phase = 'RUNNING', '', 'approach'

    def update(self, world, task=None):
        if task is not None and (task.kind, task.target) != (self.task.kind, self.task.target):
            self.__init__(task, world)
        if self.status != 'RUNNING':
            return self.status
        mine = world.actor(self.task.actor)
        if mine.position is None:
            self.status, self.reason = 'FAILURE', 'missing-own-position'
        elif (mine.position == world.spawns[mine.index] and self.previous.position is not None and
              (sum(abs(a-b) for a, b in zip(mine.position, self.previous.position)) > 1 or
               mine.carrying < self.previous.carrying) and mine.returned == self.previous.returned):
            self.status, self.reason = 'FAILURE', 'respawn'
        elif (self.task.kind in ('bank', 'escape') or self.phase == 'bank') and not mine.pacman:
            self.status, self.reason = 'SUCCESS', 'home'
        elif self.task.kind == 'harvest' and mine.carrying >= self.task.load_limit > 0:
            self.phase = 'bank'
        elif self.task.kind == 'harvest' and self.task.target not in world.food:
            if mine.carrying > self.origin.carrying:
                self.phase = 'bank'
            else:
                self.status, self.reason = 'FAILURE', 'target-disappeared'
        elif self.task.kind == 'capsule' and self.task.target not in world.capsules:
            if (any(world.actor(i).scared > self.enemy_timers[i] for i in world.enemies) or
                    mine.position == self.task.target and any(world.actor(i).scared > 0 for i in world.enemies)):
                self.status, self.reason = 'SUCCESS', 'protection-active'
            else:
                self.status, self.reason = 'FAILURE', 'target-disappeared'
        elif self.task.kind == 'intercept' and not world.actor(self.task.enemy).pacman:
            self.status, self.reason = 'SUCCESS', 'intruder-gone'
        if self.status == 'RUNNING' and self.task.deadline is not None and world.timeleft <= self.task.deadline:
            self.status, self.reason = 'FAILURE', 'deadline'
        if world.timeleft != self.last_time:
            progress = (mine.position != self.previous.position or mine.carrying != self.previous.carrying or
                        mine.returned != self.previous.returned)
            self.stagnant = 0 if progress else self.stagnant + 1
            if self.stagnant >= 3 and self.task.kind not in ('guard', 'wait') and self.status == 'RUNNING':
                self.status, self.reason = 'FAILURE', 'no-progress'
            self.previous, self.last_time = mine, world.timeleft
        return self.status


"""Context-qualified mixture calibration from our own observed enemy replies."""
import math
import time



def tl_actor_state(state,index):
    source=state.get_agent_state(index)
    position=source.get_position()
    return TLActor(index,tuple(position) if position is not None else None,
                   source.is_pacman,source.num_carrying,source.num_returned,source.scared_timer)


def tl_reply_context(actor):
    return actor.pacman,actor.carrying>=5,actor.scared>0


class TLReplyModel:
    def __init__(self):
        self.weights={}
        self.counts={}
        self.log_advantage={}
        self.last_observed_action=None

    def count(self,index,context):
        return self.counts.get((index,context),0)

    def quality(self,index,context):
        return self.log_advantage.get((index,context),0.0)/max(1,self.count(index,context))

    def predict(self,index,context,base,experts):
        if not base or any(not math.isfinite(p) or p<0 for p in base.values()) or sum(base.values())<=0:
            raise ValueError('Invalid base reply probabilities')
        total=sum(base.values())
        base={a:p/total for a,p in base.items()}
        if not experts:return base
        key=index,context
        weights=self.weights.get(key,{name:1/len(experts) for name in experts})
        mass=sum(weights.get(name,0) for name in experts)
        calibrated={a:sum(weights.get(name,0)*prob.get(a,0) for name,prob in experts.items())/max(mass,1e-12)
                    for a in base}
        strength=self.count(index,context)/(20+self.count(index,context))
        result={a:.1/len(base)+.9*((1-strength)*base[a]+strength*calibrated[a]) for a in base}
        total=sum(result.values())
        return {a:p/total for a,p in result.items()}

    def observe(self,index,context,before,after,elapsed,experts,expected_gap=3,base=None):
        if (elapsed!=expected_gap or before.position is None or after.position is None or
                before.pacman!=after.pacman or (before.scared>0)!=(after.scared>0) or
                after.scared!=max(0,before.scared-1) or
                before.carrying-after.carrying-(after.returned-before.returned)>0 or not experts):
            return False
        delta=tuple(b-a for a,b in zip(before.position,after.position))
        action={(1,0):'East',(-1,0):'West',(0,1):'North',(0,-1):'South',(0,0):'Stop'}.get(delta)
        if action is None or not any(action in p for p in experts.values()):return False
        key=index,context
        if base is not None:
            predicted=self.predict(index,context,base,experts)
            gain=math.log(max(1e-12,predicted.get(action,0)))-math.log(max(1e-12,base.get(action,0)))
            self.log_advantage[key]=self.log_advantage.get(key,0)+gain
        old=self.weights.get(key,{name:1/len(experts) for name in experts})
        updated={name:old.get(name,1/len(experts))*max(1e-6,p.get(action,0)) for name,p in experts.items()}
        total=sum(updated.values())
        self.weights[key]={name:.99*w/total+.01/len(updated) for name,w in updated.items()}
        self.counts[key]=self.counts.get(key,0)+1
        self.last_observed_action=action
        return True


def tl_reply_experts(agent,state,index,deadline):
    enemy=state.get_agent_state(index)
    if enemy.get_position() is None or time.monotonic()>=deadline:return {}
    legal=state.get_legal_actions(index)
    food=(state.get_blue_food() if state.is_on_red_team(index) else state.get_red_food()).as_list()
    home=agent.tl_geometry.enemy_border or (state.get_initial_agent_position(index),)
    observed_enemy_origins=[state.get_agent_position(i) for i in agent.get_opponents(state)
                            if state.get_agent_position(i) is not None]
    visible_ours=[]
    for i in agent.get_team(state):
        actor=state.get_agent_state(i)
        point=actor.get_position()
        if point is not None and any(abs(point[0]-p[0])+abs(point[1]-p[1])<=5 for p in observed_enemy_origins):
            visible_ours.append(actor)
    prey=[a.get_position() for a in visible_ours if a.is_pacman]
    danger=[a.get_position() for a in visible_ours if not a.is_pacman and a.scared_timer==0]
    if not enemy.is_pacman and enemy.scared_timer>0:danger=prey
    values={name:{} for name in ('food','bank','chase','evade','patrol')}
    for action in legal:
        if time.monotonic()>=deadline:return {}
        future=state.generate_successor(index,action)
        after=future.get_agent_state(index)
        point=after.get_position()
        distance_food=min((agent.distance(point,p) for p in food),default=0)
        distance_home=min(agent.distance(point,p) for p in home)
        distance_prey=min((agent.distance(point,p) for p in prey),default=distance_home)
        distance_danger=min((agent.distance(point,p) for p in danger),default=0)
        returned=after.num_returned-enemy.num_returned
        pickup=after.num_carrying-enemy.num_carrying+returned
        stop=.15 if action=='Stop' else 0
        died=(after.num_carrying<enemy.num_carrying and returned==0) or (
            point==state.get_initial_agent_position(index) and agent.distance(enemy.get_position(),point)>2)
        penalty=20 if died else 0
        values['food'][action]=-distance_food+3*pickup-stop-penalty
        values['bank'][action]=-distance_home+10*returned-stop-penalty
        values['chase'][action]=-distance_prey-stop-penalty
        values['evade'][action]=distance_danger-.2*distance_home-stop-penalty
        values['patrol'][action]=-.5*distance_home-stop-penalty
    result={}
    for name,scores in values.items():
        maximum=max(scores.values())
        weights={a:math.exp(max(-12,v-maximum)) for a,v in scores.items()}
        total=sum(weights.values())
        result[name]={a:.02/len(legal)+.98*w/total for a,w in weights.items()}
    return result


"""Independent task-value schema and dependency-free runtime inference."""
import json
import math
from pathlib import Path


TL_TASK_FEATURES = ('bias','utility','reward','duration','risk','bank','intercept',
                    'capsule','harvest','score','time_fraction','own_cargo','enemy_cargo')


def tl_finite_number(value):
    return type(value) in (int,float) and math.isfinite(value)


def tl_validate_task_model(model,expected_kind,require_fitted=True):
    if (expected_kind not in ('qlearning','evolution','neural') or model.get('kind')!=expected_kind or
            model.get('feature_schema_version')!=1 or
            model.get('feature_schema')!=list(TL_TASK_FEATURES) or
            model.get('feature_count')!=len(TL_TASK_FEATURES)):
        raise ValueError('Task model kind/feature schema mismatch')
    training=model.get('training',{})
    if require_fitted and (training.get('status')!='fitted' or
            type(training.get('updates')) is not int or training['updates']<=0):
        raise ValueError('Task model has no completed fitting evidence')
    if expected_kind=='neural':
        hidden=len(model.get('b1',[]))
        rows=model.get('w1',[])
        if (not 1<=hidden<=128 or len(rows)!=hidden or len(model.get('w2',[]))!=hidden or
                any(len(row)!=len(TL_TASK_FEATURES) for row in rows) or
                not all(tl_finite_number(x) for row in rows for x in row) or
                not all(tl_finite_number(x) for x in model['b1']+model['w2']+[model.get('b2')])):
            raise ValueError('Invalid task neural parameters')
    else:
        weights=model.get('weights',[])
        if len(weights)!=len(TL_TASK_FEATURES) or not all(tl_finite_number(x) for x in weights):
            raise ValueError('Invalid task linear parameters')


def tl_read_task_model(path,expected_kind):
    with Path(path).open() as handle:model=json.load(handle)
    tl_validate_task_model(model,expected_kind)
    return model


def tl_predict_task(model,features):
    tl_validate_task_model(model,model.get('kind'),require_fitted=False)
    if len(features)!=len(TL_TASK_FEATURES) or not all(tl_finite_number(x) for x in features):
        raise ValueError('Invalid task feature vector')
    if model['kind']=='neural':
        hidden=[math.tanh(sum(x*w for x,w in zip(features,row))+bias)
                for row,bias in zip(model['w1'],model['b1'])]
        return sum(x*w for x,w in zip(hidden,model['w2']))+model['b2']
    return sum(x*w for x,w in zip(features,model['weights']))


"""Connect joint task choice, execution and inherited family tactical planners."""
import math
import time



class TLTeamControllerMixin:
    def register_initial_state(self, state):
        super().register_initial_state(state)
        if not hasattr(self.shared, 'tl_geometry'):
            self.shared.tl_geometry = TLGeometry.from_agent(self)
            self.shared.tl_intents = {}
        self.tl_geometry = self.shared.tl_geometry
        self.tl_execution = None
        self.tl_root_world = None
        self.tl_last_abort_reason = ''
        self.tl_last_plan = None
        self.tl_cooldowns = {}

    def tl_select_plan(self, plans, state, world, deadline):
        return max(plans, key=lambda p: (p.utility, tuple((t.kind, t.target) for t in p.tasks)))

    def tl_invalidate_plan(self, reason):
        self.tl_last_abort_reason = reason
        self.shared.tl_intents.clear()

    def tl_execution_goal(self):
        execution = self.tl_execution
        task = execution.task
        if execution.phase == 'bank' and task.exit is not None:
            return 'return', task.exit
        role = {'harvest':'collect', 'bank':'return', 'escape':'escape', 'intercept':'defend',
                'guard':'guard', 'capsule':'capsule', 'wait':'guard'}[task.kind]
        return role, task.target

    def choose_action(self, state):
        action = super().choose_action(state)
        execution = self.tl_execution
        if (execution is not None and execution.status == 'RUNNING' and
                (self.role, self.goal) != self.tl_execution_goal()):
            execution.status, execution.reason = 'FAILURE', 'legacy-goal-change'
            self.tl_invalidate_plan('legacy-goal-change')
        return action

    def choose_goal(self, state):
        self.tl_last_abort_reason = ''
        world = TLWorld.from_agent(self, state)
        self.tl_root_world = world
        planning_deadline = min(self.deadline, time.monotonic() + min(.025, max(.006, .3*self.budget)))
        if self.tl_execution is not None:
            status = self.tl_execution.update(world)
            if status != 'RUNNING':
                failed = self.tl_execution.task
                if status == 'FAILURE':
                    self.tl_cooldowns[failed.kind, failed.target] = world.timeleft - 12
                self.tl_invalidate_plan(self.tl_execution.reason)
                self.tl_execution = None
        choices = {}
        for index in world.team:
            candidates = tl_candidates(world, self.tl_geometry, index,
                risk=lambda point, actor: self.risk_at(point, state=state, index=actor),
                deadline=planning_deadline)
            if not candidates:
                self.tl_invalidate_plan('planning-deadline')
                return super().choose_goal(state)
            if index == self.index:
                allowed = [task for task in candidates if world.timeleft <= self.tl_cooldowns.get(
                    (task.kind, task.target), world.timeleft)]
                candidates = tuple(allowed) or candidates
            choices[index] = candidates
        plans = tl_joint_plans(choices, world)
        if not plans or time.monotonic() >= planning_deadline:
            self.tl_invalidate_plan('planning-deadline')
            return super().choose_goal(state)
        plan = self.tl_select_plan(plans, state, world, planning_deadline)
        if plan is None:
            self.tl_invalidate_plan('planning-deadline')
            return super().choose_goal(state)
        self.tl_last_plan = plan
        for task in plan.tasks:
            self.shared.tl_intents[task.actor] = dict(task=task, at=world.timeleft)
            self.shared.targets[task.actor] = task.target
        self.shared.defending = {task.actor: task.enemy for task in plan.tasks if task.kind == 'intercept'}
        task = plan.for_actor(self.index)
        old = self.tl_execution
        if old is None or (old.task.kind, old.task.target) != (task.kind, task.target):
            self.tl_execution = TLExecution(task, world)
        return self.tl_execution_goal()

    def projected_goal(self, state):
        execution = getattr(self, 'tl_execution', None)
        if execution is None or execution.status != 'RUNNING':
            return super().projected_goal(state)
        if (self.role, self.goal) != self.tl_execution_goal():
            return super().projected_goal(state)
        task = execution.task
        if execution.phase == 'bank' and task.exit is not None:
            return task.exit
        if task.kind == 'harvest' and task.target not in self.get_food(state).as_list():
            if state.get_agent_state(self.index).num_carrying and task.exit is not None:
                return task.exit
            return super().projected_goal(state)
        if task.kind == 'capsule' and task.target not in self.get_capsules(state):
            return super().projected_goal(state)
        return task.target if task.target in self.tl_geometry.adj else super().projected_goal(state)

    def passive_action(self, state, index):
        intent = getattr(self.shared, 'tl_intents', {}).get(index)
        if intent is None or not 0 <= intent['at'] - state.data.timeleft <= 16:
            return super().passive_action(state, index)
        task = intent['task']
        target = task.target
        if target not in self.tl_geometry.adj:
            return super().passive_action(state, index)
        mine = state.get_agent_state(index)
        if task.kind == 'harvest' and mine.num_carrying >= task.load_limit > 0 and task.exit is not None:
            target = task.exit
        options = {}
        for action in state.get_legal_actions(index):
            future = state.generate_successor(index, action)
            after = future.get_agent_state(index)
            position = after.get_position()
            if position is None:
                continue
            if task.kind in ('guard', 'intercept') and not mine.is_pacman and mine.scared_timer == 0 and self.on_enemy_side(position):
                continue
            value = -self.distance(position, target) - 12*self.risk_at(position, state=future, index=index)
            value += 15*(after.num_returned - mine.num_returned)
            lost = mine.num_carrying - after.num_carrying - (after.num_returned-mine.num_returned)
            value -= 30 + 8*lost if lost > 0 else 0
            options[action] = value
        if not options:
            return super().passive_action(state, index)
        return max(options, key=lambda action: (options[action],
            action == 'Stop' if mine.get_position() == target else action != 'Stop', action))

    def terminal(self, state):
        if super().terminal(state):
            return True
        threshold = max(0, math.ceil(state.data.layout.total_food/2-2))
        return any(sum(state.get_agent_state(i).num_returned for i in team) >= threshold
                   for team in (self.get_team(state), self.get_opponents(state)))

    def tl_team_value(self, state):
        root = getattr(self, 'tl_root_world', None)
        if root is None:
            return 0.0
        value = 0.0
        for index in root.team:
            if index == self.index:
                continue
            teammate = state.get_agent_state(index)
            value += 3*teammate.num_carrying
            original = root.actor(index)
            lost = original.carrying - teammate.num_carrying - (teammate.num_returned-original.returned)
            value -= 8*max(0, lost)
        if self.role != 'defend':
            value -= 3*sum(state.get_agent_state(i).num_carrying for i in root.enemies
                           if state.get_agent_state(i).is_pacman)
        mine = state.get_agent_state(self.index)
        position = mine.get_position()
        pocket = self.tl_geometry.pocket(position)
        if mine.is_pacman and mine.num_carrying > 0 and pocket is not None:
            mouth = pocket[1]
            own_steps = self.distance(position, mouth)
            danger = 0.0
            for index in root.enemies:
                enemy = state.get_agent_state(index)
                if enemy.get_position() is None or enemy.is_pacman or enemy.scared_timer > own_steps+2:
                    continue
                distance = self.distance(enemy.get_position(), mouth)
                danger = max(danger, 3 + own_steps-distance)
            value -= (8 + 2*mine.num_carrying)*max(0, danger)
        return value

    def node_value(self, state):
        return super().node_value(state) + self.tl_team_value(state)


"""Family-specific task policies and bounded real-engine option search.

Macro adversarial/expected models are finite approximations. Inherited tactical
minimax, expectimax, UCT and MDP solvers remain separate decision backends.
"""
import math
import random
import time
from pathlib import Path



TL_FAMILIES = ('fsm','astar','minimax','expectimax','uct','belief_mcts','mdp','hierarchical',
               'qlearning','evolution','neural','adaptive','hybrid')


def tl_plan_features(agent, state, plan):
    team, enemies = agent.get_team(state), agent.get_opponents(state)
    tasks = plan.tasks
    return (1.0, plan.utility/20, sum(t.reward for t in tasks)/20,
            sum(t.duration for t in tasks)/80, sum(t.risk for t in tasks)/2,
            sum(t.kind=='bank' for t in tasks)/2,
            sum(t.kind=='intercept' for t in tasks)/2,
            sum(t.kind=='capsule' for t in tasks)/2,
            sum(t.kind=='harvest' for t in tasks)/2,
            agent.get_score(state)/30,
            state.data.timeleft/max(1, agent.shared.initial_time),
            sum(state.get_agent_state(i).num_carrying for i in team)/20,
            sum(state.get_agent_state(i).num_carrying for i in enemies)/20)


def tl_joint_value(agent, state, plan):
    value = 100*agent.get_score(state)
    own = sum(state.get_agent_state(i).num_carrying for i in agent.get_team(state))
    enemy = sum(state.get_agent_state(i).num_carrying for i in agent.get_opponents(state))
    value += 8*(own-enemy)
    for task in plan.tasks:
        actor = state.get_agent_state(task.actor)
        target = task.exit if task.kind=='harvest' and actor.num_carrying >= task.load_limit > 0 else task.target
        distance = agent.distance(actor.get_position(), target)
        value -= .8*min(100, distance)
    if agent.terminal(state):
        score = agent.get_score(state)
        value += 10000*(1 if score>0 else -1 if score<0 else 0)
    return value


def tl_option_action(agent, state, task):
    index = task.actor
    legal = state.get_legal_actions(index)
    mine = state.get_agent_state(index)
    target = task.target
    if task.kind=='harvest' and (mine.num_carrying >= task.load_limit > 0 or
                                target not in agent.get_food(state).as_list()):
        target = task.exit or min(agent.shared.border, key=lambda p:agent.distance(mine.get_position(),p))
    values = {}
    for action in legal:
        future = state.generate_successor(index, action)
        after = future.get_agent_state(index)
        position = after.get_position()
        if task.kind in ('intercept','guard') and not mine.is_pacman and mine.scared_timer==0 and agent.on_enemy_side(position):
            continue
        value = -agent.distance(position,target) - 12*agent.risk_at(position,state=future,index=index)
        value += 30*(after.num_returned-mine.num_returned)
        lost = mine.num_carrying-after.num_carrying-(after.num_returned-mine.num_returned)
        if lost>0:value -= 30+8*lost
        values[action] = value
    if not values:return 'Stop' if 'Stop' in legal else legal[0]
    return max(values,key=lambda action:(values[action],
        action=='Stop' if mine.get_position()==target else action!='Stop',action))


def tl_macro_step(agent, state, plan, plies, opponent_policy, deadline):
    if time.monotonic() >= deadline:raise TLTaskDeadline()
    if agent.terminal(state):return state.deep_copy()
    count = state.get_num_agents()
    previous = state.data._agent_moved
    index = (previous+1)%count if previous is not None else agent.index
    node = state.deep_copy()
    assignments = {task.actor:task for task in plan.tasks}
    stats = getattr(agent,'tl_task_search',None)
    for _ in range(plies):
        if time.monotonic() >= deadline:raise TLTaskDeadline()
        if agent.terminal(node):break
        if node.get_agent_position(index) is None:
            future = node.deep_copy()
            future.data.timeleft -= 1
            future.data._agent_moved = index
            item = future.get_agent_state(index)
            item.scared_timer = max(0,item.scared_timer-1)
            node = future
        elif index in assignments:
            node = node.generate_successor(index,tl_option_action(agent,node,assignments[index]))
        else:
            options = agent.opponent_actions(node,index)
            if opponent_policy=='worst':
                action = min((a for a,_ in options),key=lambda a:tl_joint_value(agent,node.generate_successor(index,a),plan))
            else:
                action = agent.tl_rng.choices([a for a,_ in options],weights=[p for _,p in options],k=1)[0]
            node = node.generate_successor(index,action)
        if stats is not None:stats['actor_steps'] += 1
        index = (index+1)%count
    return node


def tl_sample_option_world(agent, state):
    original = agent.rng
    try:
        agent.rng = agent.tl_rng
        return agent.sample_world(state)
    finally:
        agent.rng = original


def tl_option_uct(agent, state, plans, deadline, beliefs=False):
    # Nodes are histories of joint options; each edge advances all real actors.
    counts, sums = {}, {}
    root = ()
    baselines = [tl_joint_value(agent,state,p) for p in plans]
    visits = [0]*len(plans)
    while time.monotonic() < deadline:
        sampled = tl_sample_option_world(agent,state) if beliefs else state.deep_copy()
        if beliefs:agent.tl_task_search['sampled_worlds'] += 1
        node, path, result, root_choice = sampled, (), [], None
        try:
            for depth in range(2):
                if time.monotonic() >= deadline:raise TLTaskDeadline()
                untried = [i for i in range(len(plans)) if path+(i,) not in counts]
                if untried:
                    choice = untried[0]
                else:
                    total = sum(counts.get(path+(i,),0) for i in range(len(plans)))
                    choice = max(range(len(plans)),key=lambda i:
                        sums[path+(i,)]/counts[path+(i,)]+25*math.sqrt(math.log(max(2,total))/counts[path+(i,)]))
                if root_choice is None:root_choice=choice
                path += (choice,)
                node = tl_macro_step(agent,node,plans[choice],8,'expected',deadline)
                result.append(path)
                if agent.terminal(node):break
            reward = tl_joint_value(agent,node,plans[root_choice])-baselines[root_choice]+plans[root_choice].utility
        except TLTaskDeadline:
            break
        for key in result:
            counts[key]=counts.get(key,0)+1
            sums[key]=sums.get(key,0.0)+reward
        visits[root_choice] += 1
        agent.tl_task_search['iterations'] += 1
    agent.tl_task_search['root_visits'] = visits
    if not all(visits):return None
    best=max(range(len(plans)),key=lambda i:(sums[(i,)]/counts[(i,)],plans[i].utility,-i))
    return plans[best]


def tl_rule_plan(agent,state,plans):
    def priority(plan):
        value = plan.utility
        for task in plan.tasks:
            actor=state.get_agent_state(task.actor)
            if task.kind=='bank' and actor.num_carrying>=5:value += 50
            if task.kind=='intercept' and task.enemy is not None and state.get_agent_state(task.enemy).num_carrying>=5:value += 30
            if task.kind=='wait':value -= 5
        return value
    return max(plans,key=lambda p:(priority(p),p.utility))


def tl_rank_plans(family,agent,state,plans,deadline):
    if family not in TL_FAMILIES:raise ValueError('Unknown team strategy family')
    if not plans or time.monotonic()>=deadline:return None
    if not hasattr(agent,'tl_rng'):agent.tl_rng=random.Random(78001+agent.index)
    plans=tuple(sorted(plans,key=lambda p:-p.utility)[:4])
    policy=family
    if family in ('adaptive','hybrid'):
        opponents=[state.get_agent_state(i) for i in agent.get_opponents(state)]
        hidden_threat=any(a.get_position() is None and a.is_pacman and a.num_carrying>0 for a in opponents)
        close=any(a.get_position() is not None and agent.distance(state.get_agent_position(agent.index),a.get_position())<=6 for a in opponents)
        loaded=state.get_agent_state(agent.index).num_carrying>0
        if family=='hybrid' and hidden_threat:policy='belief_mcts'
        elif close:policy='minimax' if loaded else 'expectimax'
        else:policy='hierarchical'
    agent.tl_task_search=dict(policy=policy,iterations=0,actor_steps=0,sampled_worlds=0)
    if policy=='fsm':return tl_rule_plan(agent,state,plans)
    if policy=='astar':return plans[0]
    if policy=='hierarchical':
        # Preconditions were checked by task generation; prefer compatible
        # shorter decompositions after accounting for their full team value.
        return max(plans,key=lambda p:p.utility-.025*sum(t.duration for t in p.tasks))
    if policy=='mdp':
        return max(plans,key=lambda p:sum(22*t.reward*math.exp(-t.duration/40)/(t.duration+4)
            -4*t.risk for t in p.tasks)+.25*p.utility)
    if policy in ('qlearning','evolution','neural'):
        model=getattr(agent,'tl_task_model',None)
        if model is None:return plans[0]
        def learned(plan):
            features=tl_plan_features(agent,state,plan)
            value=tl_predict_task(model,features)
            return value+.1*plan.utility
        return max(plans,key=learned)
    if policy in ('uct','belief_mcts'):
        return tl_option_uct(agent,state,plans,deadline,beliefs=policy=='belief_mcts')
    # Task-level adversarial replies and sampled expected continuations are
    # finite macro models; exact inherited tactical search still follows.
    values=[]
    for plan in plans:
        estimates=[]
        try:
            baseline=tl_joint_value(agent,state,plan)
            for _ in range(1 if policy=='minimax' else 3):
                future=tl_macro_step(agent,state,plan,8,'worst' if policy=='minimax' else 'expected',deadline)
                estimates.append(tl_joint_value(agent,future,plan)-baseline)
                agent.tl_task_search['iterations'] += 1
        except TLTaskDeadline:
            return None
        values.append(plan.utility+sum(estimates)/len(estimates))
    return plans[max(range(len(plans)),key=lambda i:(values[i],plans[i].utility,-i))]


class TLFamilyMixin:
    def register_initial_state(self,state):
        self.tl_rng=random.Random(78001+self.index)
        self.tl_task_model=None
        super().register_initial_state(state)
        if not hasattr(self.shared,'tl_reply_model'):
            self.shared.tl_reply_model=TLReplyModel()
        self.tl_reply_pending=None
        self.tl_reply_cache={}
        self.tl_reply_ready=True
        if self.mode in ('qlearning','evolution','neural'):
            path=Path(__file__).with_name('task_model.json')
            if path.exists():
                self.tl_task_model=tl_read_task_model(path,self.mode)
            elif not getattr(self,'tl_offline_training',False):
                raise ValueError('Learned team requires its fitted task_model.json')

    def tl_observe_reply(self,state):
        pending=self.tl_reply_pending
        self.tl_reply_pending=None
        if pending is None:return
        after=tl_actor_state(state,pending['index'])
        self.shared.tl_reply_model.observe(pending['index'],pending['context'],pending['before'],after,
            pending['timeleft']-state.data.timeleft,pending['experts'],
            expected_gap=state.get_num_agents()-1,base=pending['base'])

    def opponent_actions(self,state,index):
        base=dict(super().opponent_actions(state,index))
        if not getattr(self,'tl_reply_ready',False):return list(base.items())
        actor=tl_actor_state(state,index)
        context=tl_reply_context(actor)
        current=getattr(self,'current',None)
        first=(self.index+1)%state.get_num_agents()
        actual_context=(current is not None and index==first and
            state.data.timeleft==current.data.timeleft-1 and state.data._agent_moved==self.index and
            current.get_agent_position(index) is not None)
        key=state.data.timeleft,hash(state),index
        cached=self.tl_reply_cache.get(key)
        experts=cached['experts'] if cached is not None else None
        if experts is None and (actual_context or self.shared.tl_reply_model.count(index,context)>0):
            experts=tl_reply_experts(self,state,index,min(getattr(self,'deadline',float('inf')),time.monotonic()+.001))
        if not experts:return list(base.items())
        if actual_context:
            self.tl_reply_cache[key]=dict(index=index,context=context,before=actor,
                timeleft=state.data.timeleft,experts=experts,base=base,state_hash=hash(state))
        result=self.shared.tl_reply_model.predict(index,context,base,experts)
        return list(result.items())

    def choose_action(self,state):
        started=time.monotonic()
        self.tl_observe_reply(state)
        self.tl_reply_cache={}
        action=super().choose_action(state)
        future=state.generate_successor(self.index,action)
        first=(self.index+1)%state.get_num_agents()
        if not self.terminal(future) and future.get_agent_position(first) is not None:
            key=future.data.timeleft,hash(future),first
            if key not in self.tl_reply_cache and time.monotonic()<started+min(.35,max(.01,self.budget))-.001:
                self.opponent_actions(future,first)
            self.tl_reply_pending=self.tl_reply_cache.get(key)
        return action

    def tl_select_plan(self,plans,state,world,deadline):
        return tl_rank_plans(self.mode,self,state,plans,deadline)


class TLVersionAgent(TLFamilyMixin, TLTeamControllerMixin, FullStateFullTurnGuardAgent):
    pass

def create_team(first_index, second_index, is_red, inference='exact', budget=0.2, num_training=0):
    if inference not in ('exact','particle'):
        raise ValueError('Unknown inference mode')
    shared = SharedTeam(inference)
    agents=[TLVersionAgent(first_index,shared,0,budget),TLVersionAgent(second_index,shared,1,budget)]
    for agent in agents:agent.mode='expectimax'
    return agents

