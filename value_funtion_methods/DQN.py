from pathlib import Path
from datetime import datetime
import sys
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root is on sys.path
# script is run from the project root or directly.
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.grid_world import GridWorld
from value_funtion_methods import FA_params as params
from src.plot_utils import plot_loss

# --- Q-network ---
class QNet(nn.Module):
    """Small MLP Q-network taking normalized (x,y) as input and outputting Q for each action."""
    def __init__(self, in_dim: int, n_actions: int, hidden_size: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
   
def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    random.seed(s)

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def choose_action(state, policy_probs):
    probs = policy_probs[state]
    actions = list(probs.keys())
    weights = [probs[a] for a in actions]
    return random.choices(actions, weights=weights, k=1)[0]

def episodes_to_dataloder(episodes):
    """Convert list of episodes (list of (state, action, reward, next_state)) into tensors."""
    states = []
    actions = []
    rewards = []
    next_states = []

    ACTION_TO_INDEX = {a: i for i, a in enumerate(GridWorld.ACTIONS.keys())}

    for s, a, r, s_next in episodes:
        states.append((s[0] / (params.GRID_SIZE - 1), s[1] / (params.GRID_SIZE - 1)))
        actions.append(ACTION_TO_INDEX[a])
        rewards.append(r)
        next_states.append((s_next[0] / (params.GRID_SIZE - 1), s_next[1] / (params.GRID_SIZE - 1)))

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)

    data_arr = (states, actions, rewards, next_states)
    dataset = TensorDataset(*data_arr)
    dataloder = DataLoader(dataset, batch_size=params.DQN_BATCH_SIZE, shuffle=True)

    return dataloder

def get_state_value_and_policy(net):
    state_value = {}
    policy = {}

    for x in range(params.GRID_SIZE):
        for y in range(params.GRID_SIZE):
            state = (x, y)
            state_tensor = torch.tensor((state[0] / (params.GRID_SIZE - 1), state[1] / (params.GRID_SIZE - 1)), dtype=torch.float32)
            q_s = net(state_tensor).squeeze(0)  # (n_actions,)
            # select best action index and its Q value
            best_idx = int(q_s.argmax().item())
            q_max_val = float(q_s[best_idx].item())
            actions_list = list(GridWorld.ACTIONS.keys())
            policy[state] = actions_list[best_idx]
            state_value[state] = q_max_val

    return state_value, policy

def train(n_actions, device, dataloder, lr, gamma, update_freq, render_freq, env, folder_path):
    in_dim = 2
    main_net = QNet(in_dim, n_actions, hidden_size=params.DQN_HIDDEN_SIZE).to(device)
    target_net = QNet(in_dim, n_actions, hidden_size=params.DQN_HIDDEN_SIZE).to(device)
    target_net.load_state_dict(main_net.state_dict())

    optimizer = torch.optim.Adam(main_net.parameters(), lr=lr)

    loss_record = []

    for ep in range(1, params.DQN_TRAINING_EPOCH + 1):
        step = 0
        for s, a_index, r, ns in dataloder:
            step += 1

            with torch.no_grad():
                q_t_s = target_net(ns)            # [batch_size, n_actions]
                q_max = q_t_s.max(dim=1).values   # [batch_size, ]
                y_t = r + gamma * q_max         # [batch_size, ]

            q_s = main_net(s)
            q_s_a = q_s.gather(1, a_index.unsqueeze(1)).squeeze(1)

            loss = F.mse_loss(y_t, q_s_a)
            loss_record.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % update_freq == 0:
                target_net.load_state_dict(main_net.state_dict())

        if ep % render_freq == 0:
            if params.SHOW_GRID_WORLD:
                state_value, policy = get_state_value_and_policy(target_net)
                env.render(state_value, policy, folder_path, 
                        title=f"epoch:{ep}/{params.DQN_TRAINING_EPOCH}"
                        )
                    
    return loss_record

def main():
    env = GridWorld(
        width=params.GRID_SIZE,
        height=params.GRID_SIZE,
        target=params.TARGET_POS,
        forbidden=params.FORBIDDEN_CELLS,
        params_module=params,
    )
   
    set_seed(params.DQN_SEED)
    device = get_device()

    n_actions = len(GridWorld.ACTIONS)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')


    behavior_probs = {}

    for x in range(env.width):
        for y in range(env.height):
            state = (x, y)
            behavior_probs[state] = {a: 1.0 / n_actions for a in env.ACTIONS.keys()}

    # create replay buffer for DQN (dataloder)
    episodes = []
    for _ in range(params.DQN_EPISODE):
        episode = env.generate_episode(params.START_POS, behavior_probs, max_length=params.DQN_EPISODE_MAX_LENGTH)
        episodes.extend(episode)

    replay_buffer = episodes_to_dataloder(episodes)
    print(f"length of replay_buffer: {len(replay_buffer)}")

    folder_path = str(project_root / "renders" / "DQN" / f"{timestamp}")
    loss_record = train(n_actions, device, replay_buffer, params.DQN_LR, params.DQN_DISCOUNT_FACTOR,
                    params.DQN_TARGET_UPDATE_FREQ, params.DQN_TRAINING_EPOCH // 10, env, folder_path)

    if params.SHOW_GRID_WORLD:            
        plot_loss(loss_record, out_dir=folder_path,
                title="DQN Loss Curve", file_name="loss_curve.png"
                )
        
if __name__ == "__main__":
    main()