from pathlib import Path
import random

project_root = Path(__file__).resolve().parent.parent

from src.grid_world import GridWorld

class QLearning:
    def __init__(self, env, alpha, gamma):
        self.env = env

        self.states = self.env.states
        self.actions = self.env.actions

        self.alpha = alpha
        self.gamma = gamma
        self.start_pos = self.env.start_state

        n_actions = len(self.actions)
        self.qvalues = {state: {action: 0.0 for action in self.actions} for state in self.states}
        self.behavior_policy_probs = {
            state: {action: 1.0 / n_actions for action in self.actions}
            for state in self.states
        }
        self.target_policy = {state: self.actions[0] for state in self.states}
        self.values = {state: 0.0 for state in self.states}

    def solve(self, n_episodes, episode_length):
        for _ in range(n_episodes):
            state_t = self.env.reset(self.start_pos)

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

        self.env.render(self.values, self.target_policy, folder_path=str(project_root / "renders" / "q_learning"),
            title=f'n_episodes={n_episodes}, '
                +f'episode_length={episode_length}, '
                +f'alpha={self.alpha}, '
                +f'gamma={self.gamma}'
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
        "gamma": 0.9,
        "n_episodes": 500,
        "episode_length": 200,
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

    ql = QLearning(
        env=env,
        alpha=config["alpha"],
        gamma=config["gamma"],
    )
    ql.solve(
        n_episodes=config["n_episodes"],
        episode_length=config["episode_length"],
    )
                        
