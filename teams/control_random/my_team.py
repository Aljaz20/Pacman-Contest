# Course controls. Framework attribution: UC Berkeley CS188, https://ai.berkeley.edu.
import random
from contest.capture_agents import CaptureAgent
class RandomControl(CaptureAgent):
    def choose_action(self, state):
        return random.choice(state.get_legal_actions(self.index))
def create_team(a, b, is_red, num_training=0):
    return [RandomControl(a), RandomControl(b)]
