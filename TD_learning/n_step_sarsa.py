from pathlib import Path
import sys
import random

# Ensure project root is on sys.path
# script is run from the project root or directly.
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.grid_world import GridWorld
from TD_learning import TD_params as params
from src.plot_utils import plot_episode_lengths

def main():
    env = GridWorld(
        width=params.GRID_SIZE,
        height=params.GRID_SIZE,
        target=params.TARGET_POS,
        forbidden=params.FORBIDDEN_CELLS,
        params_module=params,
    )

    # initialization
    Q = {}
    most_likely_action = {}
    state_value = {}
    policy_probs = {}
    episode_length = []
    
    n_actions = len(GridWorld.ACTIONS)
    n_steps = params.SARSA_N_STEPS

    for x in range(env.width):
        for y in range(env.height):
            state = (x, y)
            Q[state] = {a: 0.0 for a in env.ACTIONS.keys()} # Q[target] should be zero
            policy_probs[state] = {a: 1.0 / n_actions for a in env.ACTIONS.keys()}

    policy_probs[params.TARGET_POS] = {a: 0.0 for a in env.ACTIONS.keys()}
    policy_probs[params.TARGET_POS][5] = 1.0  # stay at target

    # n-step Sarsa
    for _ in range(params.SARSA_EPISODES):
        step = 0    # time step
        state_t = params.START_POS
        probs = policy_probs[state_t]
        actions = list(probs.keys())
        weights = [probs[a] for a in actions]
        action_t = random.choices(actions, weights=weights, k=1)[0]

        states_list = [state_t]
        actions_list = [action_t]
        rewards = [0]  # reward at time t=0 is unused

        while not env.is_target(state_t):
            step += 1
            state_t1, reward = env.get_next_state_and_reward(state_t, action_t)
            states_list.append(state_t1)
            rewards.append(reward)

            probs = policy_probs[state_t1]
            actions = list(probs.keys())
            weights = [probs[a] for a in actions]
            action_t1 = random.choices(actions, weights=weights, k=1)[0]
            actions_list.append(action_t1)

            tau = step - n_steps    # tau is the time whose estimate is being updated
            if tau >= 0:
                G = Q[states_list[step]][actions_list[step]]
                for t in reversed(range(tau + 1, step + 1)):
                    G = rewards[t] + params.SARSA_DISCOUNT_FACTOR * G

                state_tau = states_list[tau]
                action_tau = actions_list[tau]
                Q[state_tau][action_tau] += params.SARSA_ALPHA * (G - Q[state_tau][action_tau])

                # Update policy
                max_q = max(Q[state_t].values())
                best_actions = [a for a, q in Q[state_t].items() if q == max_q]
                n_best = len(best_actions)
                for a in env.ACTIONS.keys():
                    policy_probs[state_t][a] = params.SARSA_EPSILON / n_actions
                if n_best > 0:
                    add = (1.0 - params.SARSA_EPSILON) / n_best
                    for a in best_actions:
                        policy_probs[state_t][a] += add

            state_t = state_t1
            action_t = action_t1

        episode_length.append(step)

    if params.SHOW_GRID_WORLD:
        # compute most likely action and state-value function
        for x in range(env.width):
            for y in range(env.height):
                state = (x, y)
                best_action = None
                best_prob = float('-inf')
                state_value[state] = 0
                for action, prob in policy_probs[state].items():
                    state_value[state] += Q[state][action] * prob
                    if prob > best_prob:
                        best_prob = prob
                        best_action = action
                most_likely_action[state] = best_action

        env.render(state_value, most_likely_action, folder_path=str(project_root / "renders" / "n_step_sarsa"),
                   title=f'episodes={params.SARSA_EPISODES}, n={n_steps}, '
                   +f'alpha={params.SARSA_ALPHA}, '
                   +f'eps={params.SARSA_EPSILON}, '
                   +f'discount={params.SARSA_DISCOUNT_FACTOR}'
                   )
        
        # plot episode lengths over episodes
        plot_episode_lengths(
                episode_length,
                out_dir=str(project_root / "renders" / "n_step_sarsa"),
            )
        
if __name__ == "__main__":
    main()