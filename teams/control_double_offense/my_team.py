# Course controls. Framework attribution: UC Berkeley CS188, https://ai.berkeley.edu.
from contest.baseline_team import OffensiveReflexAgent
def create_team(a, b, is_red, num_training=0):
    return [OffensiveReflexAgent(a), OffensiveReflexAgent(b)]
