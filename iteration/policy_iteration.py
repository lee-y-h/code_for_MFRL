from pathlib import Path
import sys

# Ensure project root is on sys.path
# script is run from the project root or directly.
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.grid_world import GridWorld
import params

def main():
    env = GridWorld(
        width=params.GRID_SIZE,
        height=params.GRID_SIZE,
        target=params.GOAL_POS,
        forbidden=params.FORBIDDEN_CELLS,
    )

    # Initialize policy
    V = {}
    A = {}
    for x in range(env.width):
        for y in range(env.height):
            V[(x, y)] = 0
            A[(x, y)] = 1  # arbitrary initial action

    iteration_pi = params.POLICY_IMPROVEMENT_STEPS
    for it_pi in range(params.POLICY_IMPROVEMENT_STEPS):
        # Policy Evaluation
        for it_pe in range(params.POLICY_EVALUATION_STEPS):
            V_new = {}
            for x in range(env.width):
                for y in range(env.height):
                    state = (x, y)
                    action = A[state]
                    next_state, reward = env.get_next_state_and_reward(state, action)
                    V_new[state] = reward + params.POLICY_ITERATION_DISCOUNT_FACTOR * V.get(next_state, 0)
            
            # check for convergence
            delta = max(abs(V[s] - V_new[s]) for s in V)
            V = V_new
            if delta < params.POLICY_EVALUATION_THRESHOLD:
                break

        # Policy Improvement
        policy_stable = True
        for x in range(env.width):
            for y in range(env.height):
                state = (x, y)
                old_action = A[state]
                best_action = old_action
                best_value = float('-inf')
                for action in env.ACTIONS.keys():
                    next_state, reward = env.get_next_state_and_reward(state, action)
                    action_value = reward + params.POLICY_ITERATION_DISCOUNT_FACTOR * V[next_state]
                    if action_value > best_value:
                        best_value = action_value
                        best_action = action
                A[state] = best_action
                if old_action != best_action:
                    policy_stable = False

        if policy_stable:
            iteration_pi = it_pi
            break

    if params.SHOW_GRID_WORLD:
        env.render(V, A, folder_path=str(project_root / 'renders' / 'policy_iteration'),
                   title=f'policy_improvement_steps={iteration_pi}, '
                   +f'policy_evaluation_steps={params.POLICY_EVALUATION_STEPS}, '
                   +f'r_target={params.REWARD_TARGET}, '
                   +f'r_forbidden={params.REWARD_FORBIDDEN}, '
                   +f'discount={params.POLICY_ITERATION_DISCOUNT_FACTOR}'
                   )
        
if __name__ == "__main__":
    main()