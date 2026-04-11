from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

from src.grid_world import GridWorld

class MCEpsilonGreedy:
    def __init__(self, env, gamma):
        self.env = env

        self.states = self.env.states
        self.actions = self.env.actions
        self.n_actions = len(self.actions)

        self.gamma = gamma

        self.avg_return = {(state, action): 0.0 for state in self.states for action in self.actions}
        self.return_counts = {(state, action): 0 for state in self.states for action in self.actions}
        self.policy = {state: self.actions[0] for state in self.states}  # greedy action under epsilon-greedy policy
        self.policy_probs = {state: {} for state in self.states}  # action probabilities per state
        self.values = {state: 0.0 for state in self.states}

    def solve(self, n_episodes, episode_length, epsilon):
        self.policy_probs = {
            state: {action: 1.0 / self.n_actions for action in self.actions}
            for state in self.states
        }

        for _ in range(n_episodes):
            # Start each episode with a random state-action pair to ensure exploration
            state, action = self.env.sample_state_action_pair()
            episode = self.env.generate_stochastic_episode(state, self.policy_probs, episode_length, action=action)

            # Calculate returns and update action-value estimates
            discounted_return = 0.0
            for state_t, action_t, reward_t, _, _ in reversed(episode):
                discounted_return = self.gamma * discounted_return + reward_t
                self.return_counts[(state_t, action_t)] += 1
                alpha = 1.0 / self.return_counts[(state_t, action_t)]
                self.avg_return[(state_t, action_t)] += (discounted_return - self.avg_return[(state_t, action_t)]) * alpha

            # Update epsilon-greedy policy
            for state_t in self.states:
                visited_actions = [a for a in self.actions if self.return_counts[(state_t, a)] > 0]
                if not visited_actions:
                    continue

                qvalues = {a: self.avg_return[(state_t, a)] for a in visited_actions}
                self.policy[state_t] = max(qvalues, key=lambda act: qvalues[act])

                uniform_prob = epsilon / self.n_actions
                self.policy_probs[state_t] = {a: uniform_prob for a in self.actions}
                self.policy_probs[state_t][self.policy[state_t]] += (1.0 - epsilon)

        # Calculate state values under the final epsilon-greedy policy
        for state in self.states:
            state_value = 0.0
            for action, prob in self.policy_probs[state].items():
                if self.return_counts[(state, action)] == 0:
                    continue
                state_value += self.avg_return[(state, action)] * prob
            self.values[state] = state_value

        self.env.render(self.values, self.policy, folder_path=str(project_root / "renders" / "mc_epsilon_greedy"),
            title=f'n_episodes={n_episodes}, '
            +f'episode_length={episode_length}, '
            +f'epsilon={epsilon}, '
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
        "r_step": 0,
        "r_stay": -0.2,
        "gamma": 0.9,
        "n_episodes": 10000,
        "episode_length": 200,
        "epsilon": 0.1,
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

    mc = MCEpsilonGreedy(
        env=env,
        gamma=config["gamma"],
    )
    mc.solve(
        n_episodes=config["n_episodes"],
        episode_length=config["episode_length"],
        epsilon=config["epsilon"],
    )