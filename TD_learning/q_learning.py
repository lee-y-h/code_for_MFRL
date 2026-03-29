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

class QLearning:
    def __init__(self):
        self.env = GridWorld(width=params.GRID_SIZE, height=params.GRID_SIZE,
            target=params.TARGET_POS, forbidden=params.FORBIDDEN_CELLS, params_module=params)

        self.states = self.env.states
        self.actions = self.env.actions

        self.alpha = params.Q_LEARNING_ALPHA
        self.gamma = params.Q_LEARNING_DISCOUNT_FACTOR

        n_actions = len(self.actions)
        self.qvalues = {state: {action: 0.0 for action in self.actions} for state in self.states}
        self.behavior_policy_probs = {
            state: {action: 1.0 / n_actions for action in self.actions}
            for state in self.states
        }
        self.target_policy = {state: self.actions[0] for state in self.states}
        self.values = {state: 0.0 for state in self.states}

    def solve(self, episodes, episode_length):
        for _ in range(episodes):
            state_t = self.env.reset(params.START_POS)

            for _ in range(episode_length):
                probs = self.behavior_policy_probs[state_t]
                actions = list(probs.keys())
                weights = [probs[action] for action in actions]
                action_t = random.choices(actions, weights=weights, k=1)[0]

                state_t1, reward, _ = self.env.step(action_t)

                best_next_q = max(self.qvalues[state_t1][action] for action in self.actions)
                self.qvalues[state_t][action_t] += self.alpha * (
                    reward + self.gamma * best_next_q - self.qvalues[state_t][action_t]
                )

                self.target_policy[state_t] = max(self.qvalues[state_t], key=lambda action: self.qvalues[state_t][action])
                state_t = state_t1

        for state in self.states:
            self.values[state] = self.qvalues[state][self.target_policy[state]]

        if params.SHOW_GRID_WORLD:
            self.env.render(self.values, self.target_policy, folder_path=str(project_root / "renders" / "q_learning"),
                title=f'episodes={params.Q_LEARNING_EPISODES}, '
                    +f'episode_length={params.Q_LEARNING_EPISODE_LENGTH}, '
                    +f'alpha={params.Q_LEARNING_ALPHA}, '
                    +f'discount={params.Q_LEARNING_DISCOUNT_FACTOR}'
            )


if __name__ == "__main__":
    ql = QLearning()
    ql.solve(
        episodes=params.Q_LEARNING_EPISODES,
        episode_length=params.Q_LEARNING_EPISODE_LENGTH,
    )
                        
