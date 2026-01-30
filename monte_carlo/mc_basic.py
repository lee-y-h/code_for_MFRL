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

    # initialization
    total_return = {}
    return_counts = {}
    A = {}  # policy
    action_value = {}
    for x in range(env.width):
        for y in range(env.height):
            state = (x, y)
            action_value[state] = float('-inf')
            A[state] = 1  # arbitrary initial action
            for action in env.ACTIONS.keys():
                total_return[(state, action)] = 0
                return_counts[(state, action)] = 0

    # Monte Carlo Basic
    iterations = params.MC_VALUE_ESTIMATION_MAX_ITERATE_STEPS
    for it in range(params.MC_VALUE_ESTIMATION_MAX_ITERATE_STEPS):
        old_state_value = action_value.copy()   # state_value = action_value under this policy
        old_policy = A.copy()
        for x in range(env.width):
            for y in range(env.height):
                state = (x, y)
                best_action = None
                best_action_value = float('-inf')
                for action in env.ACTIONS.keys():
                    for episode in range(params.MC_BASIC_EPISODES):
                        current_state = state
                        current_action = action
                        G = 0
                        rate = 1.0
                        for step in range(params.MC_BASIC_EPISODE_LENGTH):
                            next_state, g = env.get_next_state_and_reward(current_state, current_action)
                            G += rate * g
                            rate *= params.MC_BASIC_DISCOUNT_FACTOR
                            current_state = next_state
                            current_action = old_policy[current_state]
                        
                        total_return[(state, action)] += G
                        return_counts[(state, action)] += 1
                
                    if return_counts[(state, action)] > 0:
                        estimated_action_value = total_return[(state, action)] / return_counts[(state, action)]
                        if estimated_action_value > best_action_value:
                            best_action_value = estimated_action_value
                            best_action = action

                A[state] = best_action
                action_value[state] = best_action_value
        
        # check for convergence
        if old_policy == A:
            delta = max(abs(old_state_value[s] - action_value[s]) for s in action_value)
            if delta < params.MC_VALUE_ESTIMATION_THRESHOLD:
                iterations = it
                break
    
    if params.SHOW_GRID_WORLD:
        env.render(action_value, A, folder_path=str(project_root / 'renders' / 'mc_basic'),
                   title=f'iteration={iterations}, '
                   +f'r_target={params.REWARD_TARGET}, '
                   +f'r_forbidden={params.REWARD_FORBIDDEN}, '
                   +f'discount={params.MC_BASIC_DISCOUNT_FACTOR}, '
                   +f'episode_length={params.MC_BASIC_EPISODE_LENGTH}'
                   )
        
if __name__ == "__main__":
    main()