from pathlib import Path
import sys

# Ensure project root is on sys.path
# script is run from the project root or directly.
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.grid_world import GridWorld
from monte_carlo import monte_carlo_params as params

class MCBasic:
    def __init__(self):
        self.env = GridWorld(width=params.GRID_SIZE, height=params.GRID_SIZE,
            target=params.GOAL_POS, forbidden=params.FORBIDDEN_CELLS, params_module=params)

        self.states = self.env.states
        self.actions = self.env.actions

        self.gamma = params.MC_BASIC_DISCOUNT_FACTOR

        self.policy = {state: self.actions[0] for state in self.states}  # Initialize policy
        self.values = {state: 0.0 for state in self.states}  # Initialize value
        self.total_return = {(state, action): 0.0 for state in self.states for action in self.actions}
        self.return_counts = {(state, action): 0 for state in self.states for action in self.actions}

    def calculate_returns(self, episode):
        discounted_return = 0.0
        for _, _, reward_t, _, _ in reversed(episode):
            discounted_return = self.gamma * discounted_return + reward_t
        return discounted_return

    def solve(self, max_iterations, threshold, episodes, episode_length):
        iterations = max_iterations
        for it in range(1, max_iterations + 1):
            old_values = self.values.copy()
            old_policy = self.policy.copy()

            # Generate episodes starting from all state-action pairs
            for state in self.states:
                qvalues = {action: float('-inf') for action in self.actions}
                for action in self.actions:
                    for _ in range(episodes):
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

        if params.SHOW_GRID_WORLD:
            self.env.render(self.values, self.policy, folder_path=str(project_root / 'renders' / 'mc_basic'),
                title=f'iteration={iterations}, '
                +f'r_target={params.REWARD_TARGET}, '
                +f'r_forbidden={params.REWARD_FORBIDDEN}, '
                +f'discount={params.MC_BASIC_DISCOUNT_FACTOR}, '
                +f'episode_length={params.MC_BASIC_EPISODE_LENGTH}'
            )


if __name__ == "__main__":
    mc = MCBasic()
    mc.solve(
        max_iterations=params.MC_VALUE_ESTIMATION_MAX_ITERATE_STEPS,
        threshold=params.MC_VALUE_ESTIMATION_THRESHOLD,
        episodes=params.MC_BASIC_EPISODES,
        episode_length=params.MC_BASIC_EPISODE_LENGTH,
    )