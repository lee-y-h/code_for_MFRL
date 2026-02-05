from pathlib import Path
import sys
import random

# Ensure project root is on sys.path
# script is run from the project root or directly.
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.grid_world import GridWorld
from monte_carlo import monte_carlo_params as params

def main():
    env = GridWorld(
        width=params.GRID_SIZE,
        height=params.GRID_SIZE,
        target=params.GOAL_POS,
        forbidden=params.FORBIDDEN_CELLS,
        params_module=params,
    )

    # initialization
    avg_return = {}
    return_counts = {}
    most_likely_action = {}  # most likely action under current policy (greedy)
    policy_probs = {}  # distribution over actions per state

    n_actions = len(GridWorld.ACTIONS)
    for x in range(env.width):
        for y in range(env.height):
            state = (x, y)
            # uniform initial policy: each action has equal probability
            policy_probs[state] = {a: 1.0 / n_actions for a in env.ACTIONS.keys()}
            for action in env.ACTIONS.keys():
                avg_return[(state, action)] = 0
                return_counts[(state, action)] = 0
    
    # Monte Carlo Epsilon-Greedy
    eps = params.MC_EG_EPSILON
    iterations = params.MC_EG_MAX_ITERATE_STEPS
    for it in range(params.MC_EG_MAX_ITERATE_STEPS):
        # copy old policy probabilities for convergence test
        old_policy = {s: policy_probs[s].copy() for s in policy_probs}
        for _ in range(params.MC_EG_EPISODES):
            episode = []
            state, action = env.sample_state_action_pair()
            next_state, reward = env.get_next_state_and_reward(state, action)
            episode.append((state, action, reward))
            current_state = next_state
            for _ in range(params.MC_EG_EPISODE_LENGTH - 1):
                # sample action directly from the current policy probabilities
                probs = policy_probs[current_state]
                actions = list(probs.keys())
                weights = [probs[a] for a in actions]
                current_action = random.choices(actions, weights=weights, k=1)[0]
                next_state, reward = env.get_next_state_and_reward(current_state, current_action)
                episode.append((current_state, current_action, reward))
                current_state = next_state
            
            # Every-visit strategy
            G = 0
            for state, action, reward in reversed(episode):
                G = params.MC_EG_DISCOUNT_FACTOR * G + reward
                return_counts[(state, action)] += 1
                alpha = 1 / return_counts[(state, action)]
                avg_return[(state, action)] += (G - avg_return[(state, action)]) * alpha
        
            # Update policy based on average returns
            for x in range(env.width):
                for y in range(env.height):
                    state = (x, y)
                    best_action = None
                    best_value = float('-inf')
                    for action in env.ACTIONS.keys():
                        if return_counts[(state, action)] != 0 and avg_return[(state, action)] > best_value:
                            best_value = avg_return[(state, action)]
                            best_action = action
                    if best_action is not None:
                        most_likely_action[state] = best_action
                        # construct epsilon-greedy distribution from greedy action
                        uniform_prob = eps / n_actions
                        policy_probs[state] = {a: uniform_prob for a in env.ACTIONS.keys()}
                        policy_probs[state][most_likely_action[state]] += (1.0 - eps)

        # convergence test on policy probabilities
        if policy_probs == old_policy:
            iterations = it
            break

    # calculate state value
    state_values = {}
    for x in range(env.width):
        for y in range(env.height):
            state = (x, y)
            state_values[state] = 0
            # expected value under learned policy probabilities; skip unvisited actions
            for action, prob in policy_probs[state].items():
                if return_counts.get((state, action), 0) == 0:
                    continue
                state_values[state] += avg_return[(state, action)] * prob

    if params.SHOW_GRID_WORLD:
        env.render(state_values, most_likely_action, folder_path=str(project_root / "renders" / "mc_epsilon_greedy"),
                   title=f'iterations={iterations}, '
                   +f'episode={params.MC_EG_EPISODES}, '
                   +f'episode_length={params.MC_EG_EPISODE_LENGTH}, '
                   +f'epsilon={params.MC_EG_EPSILON}, '
                   +f'r_forbidden={params.REWARD_FORBIDDEN}, '
                   +f'r_target={params.REWARD_TARGET}, '
                   +f'discount={params.MC_EG_DISCOUNT_FACTOR}')
        
if __name__ == "__main__":
    main()