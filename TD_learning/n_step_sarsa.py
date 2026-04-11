from pathlib import Path
import random

project_root = Path(__file__).resolve().parent.parent

from src.grid_world import GridWorld
from src.plot_utils import plot_episode_stats

class NStepSarsa:
    def __init__(self, env, alpha, gamma, epsilon, n_steps):
        self.env = env

        self.states = self.env.states
        self.actions = self.env.actions
        self.n_actions = len(self.actions)

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_steps = n_steps
        self.start_pos = self.env.start_state
        self.target_pos = self.env.target
        self.n_episodes = 0

        self.state_values = {state: 0.0 for state in self.states}
        self.qvalues = {state: {action: 0.0 for action in self.actions} for state in self.states}
        self.policy_probs = {
            state: {action: 1.0 / self.n_actions for action in self.actions}
            for state in self.states
        }
        self.policy_probs[self.target_pos] = {action: 0.0 for action in self.actions}
        self.policy_probs[self.target_pos][5] = 1.0

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

    def solve(self, n_episodes, max_steps):
        self.n_episodes = n_episodes

        for _ in range(n_episodes):
            step = 0
            reward_sum = 0.0
            state_t = self.env.reset(self.start_pos)
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

        self.env.render(self.state_values, self.policy, folder_path=str(project_root / "renders" / "n_step_sarsa"),
            title=f'n_episodes={self.n_episodes}, n={self.n_steps}, '
                +f'alpha={self.alpha}, '
                +f'eps={self.epsilon}, '
                +f'gamma={self.gamma}'
        )

        plot_episode_stats(
            self.episode_lengths,
            self.total_rewards,
            out_dir=str(project_root / "renders" / "n_step_sarsa"),
            x_label="Episode",
        )
        
if __name__ == "__main__":
    config = {
        "grid_size": 5,
        "start_pos": (0, 0),
        "target_pos": (2, 3),
        "forbidden_cells": [(1, 1), (1, 3), (1, 4), (2, 1), (2, 2), (3, 3)],
        "r_target": 1,
        "r_boundary": -1,
        "r_forbidden": -10,
        "r_step": 0,
        "r_stay": -0.1,
        "alpha": 0.1,
        "gamma": 0.99,
        "epsilon": 0.1,
        "n_steps": 1,
        "max_steps": 200,
        "n_episodes": 1000,
    }

    env = GridWorld(
        width=config["grid_size"],
        height=config["grid_size"],
        target=config["target_pos"],
        forbidden=config["forbidden_cells"],
        start=config["start_pos"],
        r_target=config["r_target"],
        r_boundary=config["r_boundary"],
        r_forbidden=config["r_forbidden"],
        r_step=config["r_step"],
        r_stay=config["r_stay"],
    )

    n_step_sarsa = NStepSarsa(
        env=env,
        alpha=config["alpha"],
        gamma=config["gamma"],
        epsilon=config["epsilon"],
        n_steps=config["n_steps"],
    )
    n_step_sarsa.solve(
        n_episodes=config["n_episodes"],
        max_steps=config["max_steps"],
    )