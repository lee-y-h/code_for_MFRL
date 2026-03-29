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
from src.plot_utils import plot_episode_stats

class ExpectedSarsa:
    def __init__(self):
        self.env = GridWorld(width=params.GRID_SIZE, height=params.GRID_SIZE,
            target=params.TARGET_POS, forbidden=params.FORBIDDEN_CELLS, params_module=params)

        self.states = self.env.states
        self.actions = self.env.actions
        self.n_actions = len(self.actions)

        self.alpha = params.SARSA_ALPHA
        self.gamma = params.SARSA_DISCOUNT_FACTOR
        self.epsilon = params.SARSA_EPSILON

        self.state_values = {state: 0.0 for state in self.states}
        self.qvalues = {state: {action: 0.0 for action in self.actions} for state in self.states}
        self.policy_probs = {
            state: {action: 1.0 / self.n_actions for action in self.actions}
            for state in self.states
        }
        self.policy_probs[params.TARGET_POS] = {action: 0.0 for action in self.actions}
        self.policy_probs[params.TARGET_POS][5] = 1.0

        self.policy = {state: self.actions[0] for state in self.states}
        self.episode_lengths = []
        self.total_rewards = []

    def _sample_action(self, state):
        probs = self.policy_probs[state]
        actions = list(probs.keys())
        weights = [probs[action] for action in actions]
        return random.choices(actions, weights=weights, k=1)[0]

    def _update_epsilon_greedy_policy(self, state):
        n_actions = len(self.actions)
        max_q = max(self.qvalues[state].values())
        best_actions = [action for action, q in self.qvalues[state].items() if q == max_q]
        n_best = len(best_actions)

        for action in self.actions:
            self.policy_probs[state][action] = self.epsilon / n_actions
        if n_best > 0:
            add = (1.0 - self.epsilon) / n_best
            for action in best_actions:
                self.policy_probs[state][action] += add

    def solve(self, episodes, max_steps):
        for _ in range(episodes):
            step = 0
            reward_sum = 0.0
            state_t = self.env.reset(params.START_POS)
            done = False

            while not done and step < max_steps:
                action_t = self._sample_action(state_t)
                state_t1, reward, done = self.env.step(action_t)
                reward_sum += reward

                probs_next = self.policy_probs[state_t1]
                expected_q = sum(self.qvalues[state_t1][action] * probs_next[action] for action in self.actions)
                self.qvalues[state_t][action_t] += self.alpha * (
                    reward + self.gamma * expected_q - self.qvalues[state_t][action_t]
                )

                self._update_epsilon_greedy_policy(state_t)
                state_t = state_t1
                step += 1

            self.episode_lengths.append(step)
            self.total_rewards.append(reward_sum)

        for state in self.states:
            self.state_values[state] = max(self.qvalues[state].values())
            self.policy[state] = max(self.qvalues[state], key=lambda action: self.qvalues[state][action])

        if params.SHOW_GRID_WORLD:
            self.env.render(self.state_values, self.policy, folder_path=str(project_root / "renders" / "expected_sarsa"),
                title=f'episodes={params.EXPECTED_SARSA_EPISODES}, '
                    +f'alpha={params.SARSA_ALPHA}, '
                    +f'epsilon={params.SARSA_EPSILON}, '
                    +f'discount={params.SARSA_DISCOUNT_FACTOR}'
            )

            plot_episode_stats(
                self.episode_lengths,
                self.total_rewards,
                out_dir=str(project_root / "renders" / "expected_sarsa"),
            )

if __name__ == "__main__":
    expected_sarsa = ExpectedSarsa()
    expected_sarsa.solve(
        episodes=params.EXPECTED_SARSA_EPISODES,
        max_steps=params.EXPECTED_SARSA_MAX_EPISODE_LENGTH,
    )

    