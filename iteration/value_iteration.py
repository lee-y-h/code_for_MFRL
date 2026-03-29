from pathlib import Path
import sys

# Ensure project root is on sys.path
# script is run from the project root or directly.
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.grid_world import GridWorld
from iteration import iteration_params as params

class ValueIteration:
    def __init__(self):
        self.env = GridWorld(width=params.GRID_SIZE, height=params.GRID_SIZE,
            target=params.GOAL_POS, forbidden=params.FORBIDDEN_CELLS, params_module=params)
        
        self.states = self.env.states
        self.actions = self.env.actions

        self.gamma = params.VALUE_ITERATION_DISCOUNT_FACTOR

        self.values = {state: 0.0 for state in self.states}  # Initialize value
        self.policy = {state: self.actions[0] for state in self.states}  # Initialize policy
        
    def solve(self, max_iterations, threshold):

        self.values = {state: 0.0 for state in self.states}  # Initialize value
        
        # Value iteration
        iterations = max_iterations
        for it in range(1, max_iterations + 1):
            v_t1 = {state: 0.0 for state in self.states}  # Initialize new value
            for state in self.states:
                qvalues = {action: 0.0 for action in self.actions}  # Initialize q-values
                for action in self.actions:
                    next_state, reward = self.env.get_next_state_and_reward(state, action)
                    qvalues[action] = reward + self.gamma * self.values[next_state]

                self.policy[state] = max(qvalues, key=lambda action: qvalues[action])
                v_t1[state] = qvalues[self.policy[state]]

            # Check for convergence
            delta = max(abs(self.values[s] - v_t1[s]) for s in self.values)
            self.values = v_t1
            if delta < threshold:
                iterations = it
                break

        if params.SHOW_GRID_WORLD:
            self.env.render(self.values, self.policy, folder_path=str(project_root / 'renders' / 'value_iteration'),
                    title=f'iteration={iterations}, '
                    +f'r_target={params.REWARD_TARGET}, '
                    +f'r_forbidden={params.REWARD_FORBIDDEN}, '
                    +f'discount={params.VALUE_ITERATION_DISCOUNT_FACTOR}'
                    )

if __name__ == "__main__":
    vi = ValueIteration()
    vi.solve(
        max_iterations=params.VALUE_ITERATION_MAX_ITERATE_STEPS,
        threshold=params.VALUE_ITERATION_THRESHOLD
        )
