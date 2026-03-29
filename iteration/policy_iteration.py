from pathlib import Path
import sys

# Ensure project root is on sys.path
# script is run from the project root or directly.
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.grid_world import GridWorld
from iteration import iteration_params as params

class PolicyIteration:
    def __init__(self):
        self.env = GridWorld(width=params.GRID_SIZE, height=params.GRID_SIZE,
            target=params.GOAL_POS, forbidden=params.FORBIDDEN_CELLS, params_module=params)

        self.states = self.env.states
        self.actions = self.env.actions

        self.gamma = params.POLICY_ITERATION_DISCOUNT_FACTOR

        self.values = {state: 0.0 for state in self.states}  # Initialize value
        self.policy = {state: self.actions[0] for state in self.states}  # Initialize policy

    def solve(self, max_improvement_steps, max_evaluation_steps, threshold):

        self.values = {state: 0.0 for state in self.states}  # Initialize value

        # Policy iteration
        improvement_iterations = max_improvement_steps
        for it_pi in range(1, max_improvement_steps + 1):
            # Policy evaluation
            for _ in range(max_evaluation_steps):
                v_t1 = {state: 0.0 for state in self.states}
                for state in self.states:
                    action = self.policy[state]
                    next_state, reward = self.env.get_next_state_and_reward(state, action)
                    v_t1[state] = reward + self.gamma * self.values[next_state]

                delta = max(abs(self.values[s] - v_t1[s]) for s in self.values)
                self.values = v_t1
                if delta < threshold:
                    break

            # Policy improvement
            policy_stable = True
            for state in self.states:
                old_action = self.policy[state]
                qvalues = {action: 0.0 for action in self.actions}
                for action in self.actions:
                    next_state, reward = self.env.get_next_state_and_reward(state, action)
                    qvalues[action] = reward + self.gamma * self.values[next_state]

                self.policy[state] = max(qvalues, key=lambda action: qvalues[action])
                if old_action != self.policy[state]:
                    policy_stable = False

            if policy_stable:
                improvement_iterations = it_pi
                break

        if params.SHOW_GRID_WORLD:
            self.env.render(self.values, self.policy, folder_path=str(project_root / 'renders' / 'policy_iteration'),
                    title=f'policy_improvement_steps={improvement_iterations}, '
                    +f'policy_evaluation_steps={max_evaluation_steps}, '
                    +f'r_target={params.REWARD_TARGET}, '
                    +f'r_forbidden={params.REWARD_FORBIDDEN}, '
                    +f'discount={params.POLICY_ITERATION_DISCOUNT_FACTOR}'
                    )


if __name__ == "__main__":
    pi = PolicyIteration()
    pi.solve(
        max_improvement_steps=params.POLICY_IMPROVEMENT_STEPS,
        max_evaluation_steps=params.POLICY_EVALUATION_STEPS,
        threshold=params.POLICY_EVALUATION_THRESHOLD
    )