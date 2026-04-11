from pathlib import Path
import random

project_root = Path(__file__).resolve().parent.parent

from src.grid_world import GridWorld
from src.plot_utils import plot_episode_stats

class Sarsa:
    def __init__(self, env, alpha, gamma, epsilon):
        self.env = env

        self.states = self.env.states
        self.actions = self.env.actions
        n_actions = len(self.actions)

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.start_pos = self.env.start_state
        self.target_pos = self.env.target
        self.n_episodes = 0

        self.state_values = {state: 0.0 for state in self.states}
        self.qvalues = {state: {action: 0.0 for action in self.actions} for state in self.states}
        self.policy_probs = {
            state: {action: 1.0 / n_actions for action in self.actions}
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
            done = False

            while not done and step < max_steps:
                state_t1, reward, done = self.env.step(action_t)
                reward_sum += reward

                if done:
                    td_target = reward
                    self.qvalues[state_t][action_t] += self.alpha * (td_target - self.qvalues[state_t][action_t])
                    self._update_epsilon_greedy_policy(state_t)
                    step += 1
                    break

                action_t1 = self._sample_action(state_t1)
                td_target = reward + self.gamma * self.qvalues[state_t1][action_t1]
                self.qvalues[state_t][action_t] += self.alpha * (td_target - self.qvalues[state_t][action_t])

                self._update_epsilon_greedy_policy(state_t)
                state_t = state_t1
                action_t = action_t1
                step += 1

            self.episode_lengths.append(step)
            self.total_rewards.append(reward_sum)

        for state in self.states:
            self.state_values[state] = max(self.qvalues[state].values())
            self.policy[state] = max(self.qvalues[state], key=lambda action: self.qvalues[state][action])

        self.env.render(self.state_values, self.policy, folder_path=str(project_root / "renders" / "sarsa"),
            title=f'n_episodes={self.n_episodes}, '
                +f'alpha={self.alpha}, '
                +f'epsilon={self.epsilon}, '
                +f'gamma={self.gamma}'
        )

        plot_episode_stats(
            self.episode_lengths,
            self.total_rewards,
            out_dir=str(project_root / "renders" / "sarsa"),
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
        "n_episodes": 1000,
        "max_steps": 200,
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

    sarsa = Sarsa(
        env=env,
        alpha=config["alpha"],
        gamma=config["gamma"],
        epsilon=config["epsilon"],
    )
    sarsa.solve(
        n_episodes=config["n_episodes"], 
        max_steps=config["max_steps"],
        )

    