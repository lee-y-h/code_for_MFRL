from pathlib import Path
project_root = Path(__file__).resolve().parent.parent

from src.grid_world import GridWorld

class PolicyIteration:
    def __init__(
        self,
        env,
        gamma,
    ):
        self.env = env

        self.states = self.env.states
        self.actions = self.env.actions

        self.gamma = gamma

        self.values = {state: 0.0 for state in self.states}  # Initialize value
        self.policy = {state: self.actions[0] for state in self.states}  # Initialize policy

    def solve(self, max_improvement_steps, max_evaluation_steps, threshold):

        self.values = {state: 0.0 for state in self.states}  # Initialize value

        # Policy iteration
        improvement_iterations = max_improvement_steps
        for it_pi in range(1, max_improvement_steps + 1):
            # Policy evaluation
            evaluation_converged = False
            for _ in range(max_evaluation_steps):
                v_t1 = {state: 0.0 for state in self.states}
                for state in self.states:
                    action = self.policy[state]
                    next_state, reward = self.env.get_next_state_and_reward(state, action)
                    v_t1[state] = reward + self.gamma * self.values[next_state]

                delta = max(abs(self.values[s] - v_t1[s]) for s in self.values)
                self.values = v_t1
                if delta < threshold:
                    evaluation_converged = True
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

            # Avoid early stop when policy appears stable under under-evaluated values.
            if policy_stable and evaluation_converged:
                improvement_iterations = it_pi
                break

        self.env.render(self.values, self.policy, folder_path=str(project_root / 'renders' / 'policy_iteration'),
                title=f'policy_improvement_steps={improvement_iterations}, '
                +f'policy_evaluation_steps={max_evaluation_steps}, '
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
        "r_stay": 0,
        "gamma": 0.9,
        "threshold": 1e-4,
        "max_evaluation_steps": 20,
        "max_improvement_steps": 20,
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

    pi = PolicyIteration(
        env=env,
        gamma=config["gamma"],
    )
    pi.solve(
        max_improvement_steps=config["max_improvement_steps"],
        max_evaluation_steps=config["max_evaluation_steps"],
        threshold=config["threshold"],
    )