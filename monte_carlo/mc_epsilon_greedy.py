from pathlib import Path
import sys
import random

# Ensure project root is on sys.path
# script is run from the project root or directly.
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.grid_world import GridWorld
from monte_carlo import monte_carlo_params as params

class MCEpsilonGreedy:
    def __init__(self):
        self.env = GridWorld(width=params.GRID_SIZE, height=params.GRID_SIZE,
            target=params.GOAL_POS, forbidden=params.FORBIDDEN_CELLS, params_module=params)

        self.states = self.env.states
        self.actions = self.env.actions
        self.n_actions = len(self.actions)

        self.gamma = params.MC_EG_DISCOUNT_FACTOR

        self.avg_return = {(state, action): 0.0 for state in self.states for action in self.actions}
        self.return_counts = {(state, action): 0 for state in self.states for action in self.actions}
        self.policy = {state: self.actions[0] for state in self.states}  # greedy action under epsilon-greedy policy
        self.policy_probs = {state: {} for state in self.states}  # action probabilities per state
        self.values = {state: 0.0 for state in self.states}

    def solve(self, max_iterations, episodes, episode_length, epsilon):
        self.policy_probs = {
            state: {action: 1.0 / self.n_actions for action in self.actions}
            for state in self.states
        }

        for _ in range(max_iterations):
            for _ in range(episodes):
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

        if params.SHOW_GRID_WORLD:
            self.env.render(self.values, self.policy, folder_path=str(project_root / "renders" / "mc_epsilon_greedy"),
                title=f'iterations={max_iterations}, '
                +f'episode={params.MC_EG_EPISODES}, '
                +f'episode_length={params.MC_EG_EPISODE_LENGTH}, '
                +f'epsilon={params.MC_EG_EPSILON}, '
                +f'r_forbidden={params.REWARD_FORBIDDEN}, '
                +f'r_target={params.REWARD_TARGET}, '
                +f'discount={params.MC_EG_DISCOUNT_FACTOR}'
            )


if __name__ == "__main__":
    mc = MCEpsilonGreedy()
    mc.solve(
        max_iterations=params.MC_EG_MAX_ITERATE_STEPS,
        episodes=params.MC_EG_EPISODES,
        episode_length=params.MC_EG_EPISODE_LENGTH,
        epsilon=params.MC_EG_EPSILON
    )