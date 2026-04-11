from pathlib import Path
project_root = Path(__file__).resolve().parent.parent

from src.grid_world import GridWorld

class ValueIteration:
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

        self.env.render(self.values, self.policy, folder_path=str(project_root / 'renders' / 'value_iteration'),
            title=f'iteration={iterations}, '
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
        "max_iterations": 100,
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

    vi = ValueIteration(
        env=env,
        gamma=config["gamma"],
    )
    vi.solve(
        max_iterations=config["max_iterations"],
        threshold=config["threshold"],
        )
