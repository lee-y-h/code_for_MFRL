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
    avg_return = {}
    return_counts = {}
    A = {}  # policy
    action_value = {}
    for x in range(env.width):
        for y in range(env.height):
            state = (x, y)
            action_value[state] = float('-inf')
            A[state] = 1  # arbitrary initial action
            for action in env.ACTIONS.keys():
                avg_return[(state, action)] = 0
                return_counts[(state, action)] = 0

    # Monte Carlo Exploring Starts
    for _ in range(params.MC_ES_EPISODES):
        episode = []
        state, action = env.sample_state_action_pair()
        current_state = state
        current_action = action
        for _ in range(params.MC_ES_EPISODE_LENGTH):
            next_state, reward = env.get_next_state_and_reward(current_state, current_action)
            episode.append((current_state, current_action, reward))
            current_state = next_state
            current_action = A[current_state]

        G = 0
        for t in reversed(range(len(episode))):
            state_t, action_t, reward_t = episode[t]
            G = params.MC_ES_DISCOUNT_FACTOR * G + reward_t

            # First-visit strategy
            if not any((state_t == x[0] and action_t == x[1]) for x in episode[:t]):
                return_counts[(state_t, action_t)] += 1
                alpha = 1 / return_counts[(state_t, action_t)]
                avg_return[(state_t, action_t)] += alpha * (G - avg_return[(state_t, action_t)])

                # Update action-value function
                if action_value[state_t] < avg_return[(state_t, action_t)]:
                    action_value[state_t] = avg_return[(state_t, action_t)]
                    A[state_t] = action_t

    if params.SHOW_GRID_WORLD:
        env.render(action_value, A, folder_path=str(project_root / 'renders' / 'mc_exploring_starts'),
                   title=f'episode={params.MC_ES_EPISODES}, '
                   +f'episode_length={params.MC_ES_EPISODE_LENGTH}, '
                   +f'r_target={params.REWARD_TARGET}, '
                   +f'r_forbidden={params.REWARD_FORBIDDEN}, '
                   +f'discount={params.MC_ES_DISCOUNT_FACTOR}')
        
if __name__ == "__main__":
    main()