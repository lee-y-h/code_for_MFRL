from pathlib import Path
import random
from datetime import datetime

project_root = Path(__file__).resolve().parent.parent

from src.grid_world import GridWorld
from src.plot_utils import plot_episode_stats

class QLearningWithFA:
    def __init__(self, env, gamma, alpha, epsilon):
        self.env = env

        self.states = self.env.states
        self.actions = self.env.actions
        self.grid_size = self.env.width

        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.start_pos = self.env.start_state
        self.target_pos = self.env.target

        self.feature_dim = 15
        self.w = [0.0] * (self.feature_dim * len(self.actions))
        self.policy_probs = {
            state: {a: 1.0 / len(self.actions) for a in self.actions}
            for state in self.states
        }
        self.policy_probs[self.target_pos] = {a: 0.0 for a in self.actions}
        self.policy_probs[self.target_pos][5] = 1.0

        self.policy = {state: self.actions[0] for state in self.states}
        self.values = {state: 0.0 for state in self.states}
        self.episode_lengths = []
        self.total_rewards = []

        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.folder_path = str(project_root / "renders" / "q_learning_with_FA" / f"{self.timestamp}")

    def _phi_s_a(self, state, action):
        features = []

        x, y = state
        x = x - (self.grid_size - 1) / 2
        y = y - (self.grid_size - 1) / 2
        x = x / ((self.grid_size - 1) / 2)
        y = y / ((self.grid_size - 1) / 2)

        f_state = [
            1.0,
            x,
            y,
            x**2,
            y**2,
            x * y,
            x**3,
            y**3,
            x**2 * y,
            x * y**2,
            x**4,
            y**4,
            x**3 * y,
            x**2 * y**2,
            x * y**3,
        ]
        f_zero = [0.0] * len(f_state)

        index = self.actions.index(action)
        for i, _ in enumerate(self.actions):
            if i == index:
                features.extend(f_state)
            else:
                features.extend(f_zero)

        return features

    def _q_s_a(self, state, action):
        features = self._phi_s_a(state, action)
        return sum(self.w[i] * features[i] for i in range(len(self.w)))

    def _choose_action(self, state):
        probs = self.policy_probs[state]
        actions = list(probs.keys())
        weights = [probs[a] for a in actions]
        return random.choices(actions, weights=weights, k=1)[0]

    def _update_w(self, state, action, reward, next_state):
        phi_t = self._phi_s_a(state, action)
        q_t = self._q_s_a(state, action)
        if next_state == self.target_pos:
            q_t1 = 0.0
        else:
            q_t1 = max(self._q_s_a(next_state, next_action) for next_action in self.actions)
        td_error = reward + self.gamma * q_t1 - q_t

        for i in range(len(self.w)):
            self.w[i] += self.alpha * td_error * phi_t[i]

    def _update_policy_probs(self, state):
        q_values = {a: self._q_s_a(state, a) for a in self.actions}
        max_q = max(q_values.values())
        best_actions = [a for a, q in q_values.items() if q == max_q]
        n_best = len(best_actions)
        n_actions = len(self.actions)
        add = (1.0 - self.epsilon) / n_best

        for a in self.actions:
            self.policy_probs[state][a] = self.epsilon / n_actions
            if a in best_actions:
                self.policy_probs[state][a] += add

    def solve(self, n_episodes, max_steps, log_interval=100):
        for episode in range(1, n_episodes + 1):
            step = 0
            reward_sum = 0.0
            state_t = self.env.reset(self.start_pos)
            done = False

            while step < max_steps and not done:
                action_t = self._choose_action(state_t)
                state_t1, reward, done = self.env.step(action_t)
                reward_sum += reward

                self._update_w(state_t, action_t, reward, state_t1)
                self._update_policy_probs(state_t)

                state_t = state_t1
                step += 1

            self.episode_lengths.append(step)
            self.total_rewards.append(reward_sum)

            if episode % log_interval == 0:
                for state in self.states:
                    q_values = {a: self._q_s_a(state, a) for a in self.actions}
                    self.policy[state] = max(q_values, key=lambda action: q_values[action])

                self.env.render(
                    None,
                    self.policy,
                    folder_path=self.folder_path,
                      title=f'n_episodes={n_episodes}, ' + f'alpha={self.alpha}, '
                          + f'epsilon={self.epsilon}, ',
                    file_name=f'episode_{episode}'
                )

        for state in self.states:
            q_values = {a: self._q_s_a(state, a) for a in self.actions}
            self.policy[state] = max(q_values, key=lambda action: q_values[action])
            self.values[state] = q_values[self.policy[state]]

        plot_episode_stats(
            self.episode_lengths,
            self.total_rewards,
            out_dir=self.folder_path,
            x_label="Episode",
        )


if __name__ == "__main__":
    config = {
        "grid_size": 5,
        "start_pos": (0, 0),
        "target_pos": (4, 4),
        "forbidden_cells": [(0, 2), (2, 1), (2, 3), (4, 2)],
        "r_target": 1,
        "r_boundary": -1,
        "r_forbidden": -1,
        "r_step": -0.05,
        "r_stay": -0.1,
        "gamma": 0.95,
        "alpha": 0.001,
        "epsilon": 0.1,
        "n_episodes": 1000,
        "max_steps": 200,
        "log_interval": 100,
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

    ql_fa = QLearningWithFA(
        env=env,
        gamma=config["gamma"],
        alpha=config["alpha"],
        epsilon=config["epsilon"],
    )
    ql_fa.solve(
        n_episodes=config["n_episodes"],
        max_steps=config["max_steps"],
        log_interval=config["log_interval"],
    )