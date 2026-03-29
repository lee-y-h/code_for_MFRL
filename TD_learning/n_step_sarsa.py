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

class NStepSarsa:
    def __init__(self):
        self.env = GridWorld(width=params.GRID_SIZE, height=params.GRID_SIZE,
            target=params.TARGET_POS, forbidden=params.FORBIDDEN_CELLS, params_module=params)

        self.states = self.env.states
        self.actions = self.env.actions
        self.n_actions = len(self.actions)

        self.alpha = params.SARSA_ALPHA
        self.gamma = params.SARSA_DISCOUNT_FACTOR
        self.epsilon = params.SARSA_EPSILON
        self.n_steps = params.SARSA_N_STEPS

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

    def solve(self, episodes):
        max_steps = params.SARSA_N_MAX_EPISODE_LENGTH

        for _ in range(episodes):
            step = 0
            reward_sum = 0.0
            state_t = self.env.reset(params.START_POS)
            action_t = self._sample_action(state_t)

            states_list = [state_t]
            actions_list = [action_t]
            rewards = [0.0]  # R_0 is unused; rewards are stored as R_{t+1}
            done = False

            while not done and step < max_steps:
                step += 1
                state_t1, reward, done = self.env.step(action_t)
                reward_sum += reward
                states_list.append(state_t1)
                rewards.append(reward)

                action_t1 = self._sample_action(state_t1)
                actions_list.append(action_t1)

                tau = step - self.n_steps
                if tau >= 0:
                    if not done:
                        g_return = self.qvalues[states_list[step]][actions_list[step]]
                    else:
                        g_return = 0.0

                    for t in reversed(range(tau + 1, step + 1)):
                        g_return = rewards[t] + self.gamma * g_return

                    state_tau = states_list[tau]
                    action_tau = actions_list[tau]
                    self.qvalues[state_tau][action_tau] += self.alpha * (g_return - self.qvalues[state_tau][action_tau])
                    self._update_epsilon_greedy_policy(state_tau)

                state_t = state_t1
                action_t = action_t1

            self.episode_lengths.append(step)
            self.total_rewards.append(reward_sum)

        for state in self.states:
            self.state_values[state] = max(self.qvalues[state].values())
            self.policy[state] = max(self.qvalues[state], key=lambda action: self.qvalues[state][action])

        if params.SHOW_GRID_WORLD:
            self.env.render(self.state_values, self.policy, folder_path=str(project_root / "renders" / "n_step_sarsa"),
                title=f'episodes={params.SARSA_N_EPISODES}, n={self.n_steps}, '
                    +f'alpha={params.SARSA_ALPHA}, '
                    +f'eps={params.SARSA_EPSILON}, '
                    +f'discount={params.SARSA_DISCOUNT_FACTOR}'
            )

            plot_episode_stats(
                self.episode_lengths,
                self.total_rewards,
                out_dir=str(project_root / "renders" / "n_step_sarsa"),
            )
        
if __name__ == "__main__":
    n_step_sarsa = NStepSarsa()
    n_step_sarsa.solve(episodes=params.SARSA_N_EPISODES)