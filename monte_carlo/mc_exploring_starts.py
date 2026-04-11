from pathlib import Path
project_root = Path(__file__).resolve().parent.parent

from src.grid_world import GridWorld

class MCExploringStarts:
    def __init__(self, env, gamma):
        self.env = env

        self.states = self.env.states
        self.actions = self.env.actions

        self.gamma = gamma

        self.policy = {state: self.actions[0] for state in self.states}  # Initialize policy
        self.values = {state: 0.0 for state in self.states}  # Initialize value
        self.avg_return = {(state, action): 0.0 for state in self.states for action in self.actions}
        self.return_counts = {(state, action): 0 for state in self.states for action in self.actions}

    def solve(self, n_episodes, episode_length):
        for _ in range(n_episodes):
            # Start each episode with a random state-action pair
            state, action = self.env.sample_state_action_pair()
            episode = self.env.generate_deterministic_episode(state, self.policy, episode_length, action=action)

            # First-visit strategy
            first_visit_index = {}
            for t, (state_t, action_t, _, _, _) in enumerate(episode):
                if (state_t, action_t) not in first_visit_index:
                    first_visit_index[(state_t, action_t)] = t

            discounted_return = 0.0
            for t in range(len(episode) - 1, -1, -1):
                state_t, action_t, reward_t, _, _ = episode[t]
                discounted_return = self.gamma * discounted_return + reward_t
                if first_visit_index[(state_t, action_t)] != t:
                    continue

                self.return_counts[(state_t, action_t)] += 1
                alpha = 1.0 / self.return_counts[(state_t, action_t)]
                self.avg_return[(state_t, action_t)] += alpha * (discounted_return - self.avg_return[(state_t, action_t)])

                qvalues = {action: self.avg_return[(state_t, action)] for action in self.actions}
                self.policy[state_t] = max(qvalues, key=lambda act: qvalues[act])
                self.values[state_t] = qvalues[self.policy[state_t]]

        self.env.render(self.values, self.policy, folder_path=str(project_root / 'renders' / 'mc_exploring_starts'),
            title=f'n_episodes={n_episodes}, '
            +f'episode_length={episode_length}, '
            +f'gamma={self.gamma}'
        )


if __name__ == "__main__":
    config = {
        "grid_size": 5,
        "target_pos": (2, 3),
        "forbidden_cells": [(1, 1), (1, 3), (1, 4), (2, 1), (2, 2), (3, 3)],
        "r_target": 1,
        "r_boundary": -1,
        "r_forbidden": -10,
        "r_step": -0.1,
        "r_stay": -0.2,
        "gamma": 0.9,
        "n_episodes": 10000,
        "episode_length": 200,
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

    mc = MCExploringStarts(
        env=env,
        gamma=config["gamma"],
    )
    mc.solve(
        n_episodes=config["n_episodes"],
        episode_length=config["episode_length"],
    )