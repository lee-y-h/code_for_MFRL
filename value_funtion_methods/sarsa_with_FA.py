from pathlib import Path
import sys
import random

# Ensure project root is on sys.path
# script is run from the project root or directly.
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.grid_world import GridWorld
from value_funtion_methods import FA_params as params
from src.plot_utils import plot_episode_stats

def phi_s_a(state, action):
    """
    Feature function for state-action pair (s, a).

    action-spesific phi:

    f_state = [1.0, x, y, x**2, y**2, x*y, x**3, y**3, x**2*y, x*y**2, x**4, y**4, x**3*y, x**2*y**2, x*y**3]
    f_zero = [0.0] * len(f_state)

    phi(s, a) = [f_zero, ..., f_state, ..., f_zero]
    where f_state is at the position corresponding to action a.
    """
    
    phi_s_a = []

    x, y = state
    x = x / (params.GRID_SIZE - 1) # normalize to [0, 1]
    y = y / (params.GRID_SIZE - 1) # normalize to [0, 1]

    f_state = [1.0, x, y, x**2, y**2, x*y, x**3, y**3, x**2*y, x*y**2, x**4, y**4, x**3*y, x**2*y**2, x*y**3]
    f_zero = [0.0] * len(f_state)

    index = list(GridWorld.ACTIONS.keys()).index(action)

    for i, a in enumerate(GridWorld.ACTIONS.keys()):
        if i == index:
            phi_s_a.extend(f_state)
        else:
            phi_s_a.extend(f_zero)

    return phi_s_a

def q_s_a(state, action, w):
    features = phi_s_a(state, action)
    return sum(w[i] * features[i] for i in range(len(w)))

def update_w(w, state, action, reward, next_state, next_action):
    phi_t = phi_s_a(state, action)
    q_t = q_s_a(state, action, w)
    if next_state == params.TARGET_POS:
        q_t1 = 0.0
    else:
        q_t1 = q_s_a(next_state, next_action, w)

    td_error = reward + params.SARSA_DISCOUNT_FACTOR * q_t1 - q_t

    for i in range(len(w)):
        w[i] += params.SARSA_ALPHA * td_error * phi_t[i]

def update_policy_probs(policy_probs, w, state):
    q_values = {a: q_s_a(state, a, w) for a in GridWorld.ACTIONS.keys()}
    max_q = max(q_values.values())
    best_actions = [a for a, q in q_values.items() if q == max_q]

    # Update policy probabilities for the given state
    total_best_actions = len(best_actions)
    for a in GridWorld.ACTIONS.keys():
        policy_probs[state][a] = params.SARSA_EPSILON / len(GridWorld.ACTIONS)
        if a in best_actions:
            policy_probs[state][a] += (1.0 - params.SARSA_EPSILON) / total_best_actions

def choose_action(state, policy_probs):
    probs = policy_probs[state]
    actions = list(probs.keys())
    weights = [probs[a] for a in actions]
    return random.choices(actions, weights=weights, k=1)[0]

def main():
    env = GridWorld(
        width=params.GRID_SIZE,
        height=params.GRID_SIZE,
        target=params.TARGET_POS,
        forbidden=params.FORBIDDEN_CELLS,
        params_module=params,
    )

    # initialization
    w = [0.0] * (15 * len(GridWorld.ACTIONS))  # weight vector for linear function approximation

    policy_probs = {}

    episode_length = []
    total_reward = []

    for x in range(env.width):
        for y in range(env.height):
            state = (x, y)
            policy_probs[state] = {a: 1.0 / len(GridWorld.ACTIONS) for a in env.ACTIONS.keys()}

    policy_probs[params.TARGET_POS] = {a: 0.0 for a in env.ACTIONS.keys()}
    policy_probs[params.TARGET_POS][5] = 1.0  # stay at target

    # Sarsa with function approximation
    for _ in range(params.Q_LEARNING_EPISODES):
        step = 0
        reward_sum = 0

        state_t = params.START_POS
        action_t = choose_action(state_t, policy_probs)

        while step < params.SARSA_MAX_EPISODE_LENGTH and not env.is_target(state_t):
            state_t1, reward = env.get_next_state_and_reward(state_t, action_t)
            reward_sum += reward
            
            action_t1 = choose_action(state_t1, policy_probs)

            update_w(w, state_t, action_t, reward, state_t1, action_t1)

            update_policy_probs(policy_probs, w, state_t)

            state_t, action_t = state_t1, action_t1
            step += 1

        episode_length.append(step)
        total_reward.append(reward_sum)

    if params.SHOW_GRID_WORLD:
        # calculate most likely action
        most_likely_action = {}
        for x in range(env.width):
            for y in range(env.height):
                state = (x, y)
                best_action = None
                best_prob = float('-inf')
                for action, prob in policy_probs[state].items():
                    if prob > best_prob:
                        best_prob = prob
                        best_action = action
                most_likely_action[state] = best_action

        env.render(None, most_likely_action, folder_path=str(project_root / "renders" / "sarsa_with_FA"),
                    title=f'episodes={params.Q_LEARNING_EPISODES}, '
                    +f'alpha={params.SARSA_ALPHA}, '
                    +f'epsilon={params.SARSA_EPSILON}, '
                    )
        
        plot_episode_stats(
                episode_length, 
                total_reward,
                out_dir=str(project_root / "renders" / "sarsa_with_FA")
            )
        
if __name__ == "__main__":
    main()