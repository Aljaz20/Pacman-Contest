# Course controls. Framework attribution: UC Berkeley CS188, https://ai.berkeley.edu.
from contest.capture_agents import CaptureAgent
class GreedyFood(CaptureAgent):
    def choose_action(self, state):
        position = state.get_agent_position(self.index)
        food = self.get_food(state).as_list()
        walls = state.get_walls()
        border_x = walls.width // 2 - 1 if self.red else walls.width // 2
        targets = [(border_x,y) for y in range(walls.height) if not walls[border_x][y]] if state.get_agent_state(self.index).num_carrying >= 8 or len(food) <= 2 else food
        def value(action):
            successor = state.generate_successor(self.index, action)
            pos = successor.get_agent_position(self.index)
            return -min([self.get_maze_distance(pos,p) for p in targets] or [0])
        return max(state.get_legal_actions(self.index), key=value)
def create_team(a, b, is_red, num_training=0):
    return [GreedyFood(a), GreedyFood(b)]
