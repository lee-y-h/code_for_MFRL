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
    behavior_policy_probs = {}
    target_policy = {}  # greedy policy

    n_actions = len(GridWorld.ACTIONS)

    for x in range(env.width):
        for y in range(env.height):
            state = (x, y)
            Q[state] = {a: 0.0 for a in env.ACTIONS.keys()}
            behavior_policy_probs[state] = {a: 1.0 / n_actions for a in env.ACTIONS.keys()}

    # Q-learning (off-policy)

    for _ in range(params.Q_LEARNING_EPISODES):
        state_t = params.START_POS

        for _ in range(params.Q_LEARNING_EPISODE_LENGTH):
            probs = behavior_policy_probs[state_t]
            actions = list(probs.keys())
            weights = [probs[a] for a in actions]
            action_t = random.choices(actions, weights=weights, k=1)[0]

            state_t1, reward = env.get_next_state_and_reward(state_t, action_t)

            # Update action value
            best_next_q = max(Q[state_t1][a] for a in env.ACTIONS.keys())
            Q[state_t][action_t] += params.Q_LEARNING_ALPHA * (
                reward + params.Q_LEARNING_DISCOUNT_FACTOR * best_next_q - Q[state_t][action_t]
            )

            # Update target policy 
            max_q = float('-inf')
            best_action = None
            for action, q in Q[state_t].items():
                if q > max_q:
                    max_q = q
                    best_action = action

            target_policy[state_t] = best_action

            state_t = state_t1

    if params.SHOW_GRID_WORLD:
        # calculate state values under the target policy
        state_value = {}
        for x in range(env.width):
            for y in range(env.height):
                state = (x, y)
                best_action = target_policy.get(state, None)
                if best_action is not None:
                    state_value[state] = Q[state][best_action]
                else:
                    state_value[state] = 0
        
        env.render(state_value, target_policy, folder_path=str(project_root / "renders" / "q_learning"),
                   title=f'episodes={params.Q_LEARNING_EPISODES}, '
                    +f'episode_length={params.Q_LEARNING_EPISODE_LENGTH}, '
                    +f'alpha={params.Q_LEARNING_ALPHA}, '
                    +f'discount={params.Q_LEARNING_DISCOUNT_FACTOR}'
                    )
        
if __name__ == "__main__":
    main()
                        
