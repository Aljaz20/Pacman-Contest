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

"""Public chase evidence restricted to consecutive defender/invader contexts."""
import argparse
import hashlib
import json
from pathlib import Path
import re



class RefinedSharedTeam(SharedTeam):
    def __init__(self, inference='exact'):
        super().__init__(inference)
        self.chase_context = {}
        self.chase_capsules = None

    def observe(self, agent, state):
        counts = {index: dict(values) for index, values in self.behaviour.items()}
        capsules = frozenset(agent.get_capsules(state))
        if self.chase_capsules is not None and self.chase_capsules - capsules:
            # A scared capture can reset the timer before the next observation.
            self.chase_context.clear()
        self.chase_capsules = capsules
        super().observe(agent, state)
        # Preserve belief updates while replacing the legacy unqualified counters.
        self.behaviour = counts
        tick = state.data.timeleft
        for index in self.beliefs:
            enemy = state.get_agent_state(index)
            position = enemy.get_position()
            previous = self.chase_context.pop(index, None)
            if position is None or enemy.is_pacman or enemy.scared_timer > 0:
                continue
            targets = [(agent.distance(position, state.get_agent_position(own)), own)
                       for own in agent.get_team(state)
                       if state.get_agent_state(own).is_pacman and state.get_agent_position(own) is not None]
            if not targets:
                continue
            _, target = min(targets)
            target_position = state.get_agent_position(target)
            if previous is not None:
                old_position, old_target, old_target_position, old_tick, observer = previous
                elapsed = old_tick - tick
                steps = sum((observer + offset) % state.get_num_agents() == index
                            for offset in range(elapsed)) if 0 < elapsed <= state.get_num_agents() else 0
                if (steps == 1 and old_target == target and old_position != position and
                        agent.distance(old_position, position) <= 1):
                    counts[index]['moves'] += 1
                    counts[index]['chases'] += int(agent.distance(position, old_target_position) <
                                                 agent.distance(old_position, old_target_position))
            self.chase_context[index] = (position, target, target_position, tick, agent.index)


"""Keep macro navigation while an unscared collector stages on its home side."""
import argparse
import hashlib
import json
from pathlib import Path
import re


class ContextSelectionAgent(StrategyAgent):
    def select_policy(self, state, hybrid=False):
        mine = state.get_agent_state(self.index)
        position = self.search_position(state, self.index)
        nearby_invader = any(state.get_agent_state(index).is_pacman and
            self.search_position(state, index) is not None and
            self.distance(position, self.search_position(state, index)) <= 5
            for index in self.get_opponents(state))
        if self.role == 'collect' and not mine.is_pacman and mine.scared_timer == 0 and not nearby_invader:
            self.policy = 'hierarchical'
            return self.policy
        return super().select_policy(state, hybrid)


"""Reach hidden tactical search beyond the visibility radius using legal belief mass."""
import hashlib
import json
from pathlib import Path
import re
import time


class HiddenThreatHybridAgent(ContextSelectionAgent):
    mode = 'hybrid'

    def select_policy(self, state, hybrid=False):
        self.last_hidden_trigger_mass = 0.0
        original = super().select_policy(state, hybrid)
        if not hybrid or original != 'hierarchical' or time.monotonic() >= self.deadline:
            return original
        mine = state.get_agent_state(self.index)
        if self.role == 'collect' and not mine.is_pacman and mine.scared_timer == 0:
            return original
        position = self.search_position(state, self.index)
        for index in self.get_opponents(state):
            enemy = state.get_agent_state(index)
            if self.search_position(state,index) is not None:
                continue
            relevant = (enemy.is_pacman and not self.on_enemy_side(position)) or (
                not enemy.is_pacman and enemy.scared_timer <= 2 and self.on_enemy_side(position))
            if not relevant:
                continue
            mass = 0.0
            for point, probability in self.shared.beliefs.get(index,{}).items():
                if time.monotonic() >= self.deadline:
                    return original
                if self.distance(position,point) <= 7:
                    mass += probability
            if mass >= .12:
                self.last_hidden_trigger_mass = mass
                self.policy = 'belief_mcts'
                return self.policy
        return original



"""Small public-compatible contamination floor for sampled hidden beliefs."""
import time


def public_support(agent, state, index, deadline):
    enemy = state.get_agent_state(index)
    own = [state.get_agent_position(i) for i in agent.get_team(state)]
    origin = state.get_agent_position(agent.index)
    noisy = state.get_agent_distances()[index]
    eligible = []
    for point in agent.shared.cells:
        if time.monotonic() >= deadline:
            return None
        if (((point[0] < agent.shared.width//2) == agent.red) == enemy.is_pacman and
                all(abs(point[0]-p[0])+abs(point[1]-p[1]) > 5 for p in own if p is not None) and
                state.get_distance_prob(abs(point[0]-origin[0])+abs(point[1]-origin[1]), noisy) > 0):
            eligible.append(point)
    return eligible


class SupportRecoveryMixin:
    def recover_support(self, agent, state, deadline, excluded=()):
        if self.inference != 'particle' or time.monotonic() >= deadline:
            return
        updates = {}
        for index, prior in self.beliefs.items():
            if index in excluded or state.get_agent_state(index).get_position() is not None:
                continue
            support = public_support(agent, state, index, deadline)
            if support is None:
                return
            if support:
                mass = sum(prior.get(p, 0) for p in support)
                updates[index] = ({p: (15/16)*prior.get(p, 0)/mass+(1/16)/len(support) for p in support}
                                  if mass else {p: 1/len(support) for p in support})
        if time.monotonic() < deadline:
            self.beliefs.update(updates)

    def observe(self, agent, state):
        captured = set()
        for index in self.beliefs:
            previous = self.previous_public.get(index)
            enemy = state.get_agent_state(index)
            if (previous is not None and previous[0] and previous[1] > 0 and not enemy.is_pacman and
                    not enemy.num_carrying and enemy.num_returned == previous[2]):
                captured.add(index)
        super().observe(agent, state)
        # Preserve proven capture reachability and visible singleton observations.
        self.recover_support(agent, state, getattr(agent, 'deadline', 0), captured)



class SupportRecoverySharedTeam(SupportRecoveryMixin, RefinedSharedTeam):
    pass


def create_team(first_index, second_index, is_red, inference='particle', budget=0.07, num_training=0):
    if inference not in ('exact','particle'):
        raise ValueError('Unknown inference mode')
    shared=SupportRecoverySharedTeam(inference)
    return [HiddenThreatHybridAgent(first_index,shared,0,budget),HiddenThreatHybridAgent(second_index,shared,1,budget)]
