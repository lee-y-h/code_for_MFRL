# Grid world parameters
GRID_SIZE = 5
GOAL_POS = (2, 3)
FORBIDDEN_CELLS = [(1, 1), (1, 3), (1, 4), (2, 1), (2, 2), (3, 3)]

# GRID_SIZE = 3
# GOAL_POS = (2, 2)
# FORBIDDEN_CELLS = [(0, 2), (2, 1)]

# Reward parameters
REWARD_TARGET = 1
REWARD_BOUNDARY = -1
REWARD_FORBIDDEN = -10
REWARD_STEP = 0