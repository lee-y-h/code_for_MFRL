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
    
    x = x - (params.GRID_SIZE - 1) / 2  # center the grid at (0, 0)
    y = y - (params.GRID_SIZE - 1) / 2  # center the grid at (0, 0)
    x = x / ((params.GRID_SIZE - 1) / 2)  # normalize to [-1, 1]
    y = y / ((params.GRID_SIZE - 1) / 2)  # normalize to [-1, 1]
   
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

    td_error = reward + params.Q_LEARNING_DISCOUNT_FACTOR * q_t1 - q_t

    for i in range(len(w)):
        w[i] += params.Q_LEARNING_ALPHA * td_error * phi_t[i]

def update_policy_probs(policy_probs, w, state):
    q_values = {a: q_s_a(state, a, w) for a in GridWorld.ACTIONS.keys()}
    max_q = max(q_values.values())
    best_actions = [a for a, q in q_values.items() if q == max_q]
    n_best = len(best_actions)
    n_actions = len(GridWorld.ACTIONS)
    add = (1.0 - params.Q_LEARNING_EPSILON) / n_best
    for a in GridWorld.ACTIONS.keys():
        policy_probs[state][a] = params.Q_LEARNING_EPSILON / n_actions
        if a in best_actions:
            policy_probs[state][a] += add

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
    target_policy = {}
    episode_length = []
    total_reward = []

    for x in range(env.width):
        for y in range(env.height):
            state = (x, y)
            policy_probs[state] = {a: 1.0 / len(GridWorld.ACTIONS) for a in env.ACTIONS.keys()}

    policy_probs[params.TARGET_POS] = {a: 0.0 for a in env.ACTIONS.keys()}
    policy_probs[params.TARGET_POS][5] = 1.0  # stay at target

    # Q-learning with function approximation
    for _ in range(params.Q_LEARNING_EPISODES):
        step = 0
        reward_sum = 0

        state_t = params.START_POS

        while step < params.Q_LEARNING_EPISODE_LENGTH and not env.is_target(state_t):
            action_t = choose_action(state_t, policy_probs)
            state_t1, reward = env.get_next_state_and_reward(state_t, action_t)
            reward_sum += reward
            
            q_values_next = {a: q_s_a(state_t1, a, w) for a in GridWorld.ACTIONS.keys()}
            max_q_next = max(q_values_next.values())
            best_actions_next = [a for a, q in q_values_next.items() if q == max_q_next]
            argmax_action = random.choice(best_actions_next)

            update_w(w, state_t, action_t, reward, state_t1, argmax_action)

            update_policy_probs(policy_probs, w, state_t)

            state_t = state_t1
            step += 1

        episode_length.append(step)
        total_reward.append(reward_sum)

    if params.SHOW_GRID_WORLD:
        # construct target policy and calculate state values
        state_values = {}
        for x in range(env.width):
            for y in range(env.height):
                state = (x, y)
                q_values = {a: q_s_a(state, a, w) for a in env.ACTIONS.keys()}
                best_action = None
                max_q = float('-inf')
                for a, q in q_values.items():
                    if q > max_q:
                        max_q = q
                        best_action = a
                target_policy[state] = best_action
                state_values[state] = max_q

        env.render(state_values, target_policy, folder_path=str(project_root / "renders" / "q_learning_with_FA"),
                    title=f'episodes={params.Q_LEARNING_EPISODES}, '
                    +f'alpha={params.Q_LEARNING_ALPHA}, '
                    )
        
        plot_episode_stats(
                episode_length, 
                total_reward,
                out_dir=str(project_root / "renders" / "q_learning_with_FA")
            )
        
if __name__ == "__main__":
    main()