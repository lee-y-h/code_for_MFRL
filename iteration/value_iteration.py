from pathlib import Path
import sys

# Ensure project root is on sys.path
# script is run from the project root or directly.
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.grid_world import GridWorld
import params

def main():
    env = GridWorld(
        width=params.GRID_SIZE,
        height=params.GRID_SIZE,
        target=params.GOAL_POS,
        forbidden=params.FORBIDDEN_CELLS,
    )

    # Initialize value function
    V = {}
    A = {}
    for x in range(env.width):
        for y in range(env.height):
            V[(x, y)] = 0
            A[(x, y)] = None

    # Value iteration
    iterations = params.VALUE_ITERATION_MAX_ITERATE_STEPS
    for it in range(params.VALUE_ITERATION_MAX_ITERATE_STEPS):  # arbitrary number of iterations
        V_new = {}
        for x in range(env.width):
            for y in range(env.height):
                state = (x, y)
                for action in env.ACTIONS.keys():
                    next_state, reward = env.get_next_state_and_reward(state, action)
                    if V_new.get(state, float('-inf')) < reward + params.VALUE_ITERATION_DISCOUNT_FACTOR * V[next_state]:
                        V_new[state] = reward + params.VALUE_ITERATION_DISCOUNT_FACTOR * V[next_state]
                        A[state] = action

        # Check for convergence
        delta = max(abs(V[s] - V_new[s]) for s in V)
        V = V_new
        if delta < params.VALUE_ITERATION_THRESHOLD:
            iterations = it
            break

    if params.SHOW_GRID_WORLD:
        env.render(V, A, folder_path=str(project_root / 'renders' / 'value_iteration'),
                   title=f'iteration={iterations}, '
                   +f'r_target={params.REWARD_TARGET}, '
                   +f'r_forbidden={params.REWARD_FORBIDDEN}, '
                   +f'discount={params.VALUE_ITERATION_DISCOUNT_FACTOR}'
                   )

if __name__ == "__main__":
    main()
