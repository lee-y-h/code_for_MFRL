from pathlib import Path
project_root = Path(__file__).resolve().parent.parent

from src.grid_world import GridWorld

class MCBasic:
    def __init__(self, env, gamma):
        self.env = env

        self.states = self.env.states
        self.actions = self.env.actions

        self.gamma = gamma

        self.policy = {state: self.actions[0] for state in self.states}  # Initialize policy
        self.values = {state: 0.0 for state in self.states}  # Initialize value
        self.total_return = {(state, action): 0.0 for state in self.states for action in self.actions}
        self.return_counts = {(state, action): 0 for state in self.states for action in self.actions}

    def calculate_returns(self, episode):
        discounted_return = 0.0
        for _, _, reward_t, _, _ in reversed(episode):
            discounted_return = self.gamma * discounted_return + reward_t
        return discounted_return

    def solve(self, max_iterations, threshold, n_episodes, episode_length):
        iterations = max_iterations
        for it in range(1, max_iterations + 1):
            old_values = self.values.copy()
            old_policy = self.policy.copy()

            # Generate episodes starting from all state-action pairs
            for state in self.states:
                qvalues = {action: float('-inf') for action in self.actions}
                for action in self.actions:
                    for _ in range(n_episodes):
                        episode = self.env.generate_deterministic_episode(state, self.policy, episode_length, action=action)
                        discounted_return = self.calculate_returns(episode)
                        self.total_return[(state, action)] += discounted_return
                        self.return_counts[(state, action)] += 1

                    if self.return_counts[(state, action)] > 0:
                        qvalues[action] = self.total_return[(state, action)] / self.return_counts[(state, action)]

                # policy improvement
                self.policy[state] = max(qvalues, key=lambda act: qvalues[act])
                self.values[state] = qvalues[self.policy[state]]

            # Check for convergence
            if old_policy == self.policy:
                delta = max(abs(old_values[s] - self.values[s]) for s in self.values)
                if delta < threshold:
                    iterations = it
                    break

        self.env.render(self.values, self.policy, folder_path=str(project_root / 'renders' / 'mc_basic'),
            title=f'iteration={iterations}, '
            +f'gamma={self.gamma}, '
            +f'episode_length={episode_length}'
        )


if __name__ == "__main__":
    config = {
        "grid_size": 5,
        "target_pos": (2, 3),
        "forbidden_cells": [(1, 1), (1, 3), (1, 4), (2, 1), (2, 2), (3, 3)],
        "r_target": 1,
        "r_boundary": -1,
        "r_forbidden": -10,
        "r_step": 0,
        "r_stay": -0.2,
        "gamma": 0.9,
        "max_iterations": 1000,
        "threshold": 1e-4,
        "n_episodes": 1,
        "episode_length": 20,
    }

    env = GridWorld(
        width=config["grid_size"],
        height=config["grid_size"],
        target=config["target_pos"],
        forbidden=config["forbidden_cells"],
        r_target=config["r_target"],
        r_boundary=config["r_boundary"],
        r_forbidden=config["r_forbidden"],
        r_step=config["r_step"],
        r_stay=config["r_stay"],
    )

    mc = MCBasic(
        env=env,
        gamma=config["gamma"],
    )
    mc.solve(
        max_iterations=config["max_iterations"],
        threshold=config["threshold"],
        n_episodes=config["n_episodes"],
        episode_length=config["episode_length"],
    )