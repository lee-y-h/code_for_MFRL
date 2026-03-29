from pathlib import Path
import sys

# Ensure project root is on sys.path
# script is run from the project root or directly.
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.grid_world import GridWorld
from monte_carlo import monte_carlo_params as params

class MCExploringStarts:
    def __init__(self):
        self.env = GridWorld(width=params.GRID_SIZE, height=params.GRID_SIZE,
            target=params.GOAL_POS, forbidden=params.FORBIDDEN_CELLS, params_module=params)

        self.states = self.env.states
        self.actions = self.env.actions

        self.gamma = params.MC_ES_DISCOUNT_FACTOR

        self.policy = {state: self.actions[0] for state in self.states}  # Initialize policy
        self.values = {state: 0.0 for state in self.states}  # Initialize value
        self.avg_return = {(state, action): 0.0 for state in self.states for action in self.actions}
        self.return_counts = {(state, action): 0 for state in self.states for action in self.actions}

    def solve(self, max_iterations, episodes, episode_length):
        iterations = max_iterations
        for it in range(1, max_iterations + 1):
            old_policy = self.policy.copy()

            for _ in range(episodes):
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

            # Check for convergence
            if self.policy == old_policy:
                iterations = it
                break

        if params.SHOW_GRID_WORLD:
            self.env.render(self.values, self.policy, folder_path=str(project_root / 'renders' / 'mc_exploring_starts'),
                title=f'iterations={iterations}, '
                +f'episode={params.MC_ES_EPISODES}, '
                +f'episode_length={params.MC_ES_EPISODE_LENGTH}, '
                +f'r_target={params.REWARD_TARGET}, '
                +f'r_forbidden={params.REWARD_FORBIDDEN}, '
                +f'discount={params.MC_ES_DISCOUNT_FACTOR}'
            )


if __name__ == "__main__":
    mc = MCExploringStarts()
    mc.solve(
        max_iterations=params.MC_ES_ITERATIONS,
        episodes=params.MC_ES_EPISODES,
        episode_length=params.MC_ES_EPISODE_LENGTH,
    )