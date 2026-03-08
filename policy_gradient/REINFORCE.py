from pathlib import Path
from datetime import datetime
import re
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
from policy_gradient import policy_gradient_params as params
from src.plot_utils import plot_loss

class REINFORCE:
    def __init__(self, seed) -> None:
        self.set_seed(seed)

        self.env = GridWorld(width=params.GRID_SIZE, height=params.GRID_SIZE,
            target=params.TARGET_POS, forbidden=params.FORBIDDEN_CELLS, params_module=params)
        
        self.states = self.env.states
        self.n_states = len(self.states)
        self.actions = self.env.actions
        self.n_actions = len(self.actions)

        self.gamma = params.REINFORCE_GAMMA
        self.lr = params.REINFORCE_LR

        self.device = self.get_device()
        self.PI_net = nn.Sequential(
            nn.Linear(2, params.REINFORCE_NET_SIZE),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(params.REINFORCE_NET_SIZE, self.n_actions)
        )
        self.PI_net.to(self.device)
        self.optimizer = torch.optim.Adam(self.PI_net.parameters(), lr = self.lr)

        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.folder_path = str(project_root / "renders" / "REINFORCE" / f"{self.timestamp}")

    def set_seed(self, s):
        np.random.seed(s)
        torch.manual_seed(s)
        random.seed(s)

    def get_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        return device

    def state_to_tensor(self, state):
        x_ = self.env.width - 1
        y_ = self.env.height - 1

        x, y = state
        states_tensor = torch.tensor((x / x_, y / y_), dtype=torch.float32, device=self.device)

        return states_tensor    

    def solve(self, num_episodes, episode_max_len):
        loss_record = []
        for ep in range(1, num_episodes + 1):
            state = params.START_POS
            state_tensor = self.state_to_tensor(state)

            episode = []
            done = False
            step = 0
            while not done and step < episode_max_len:
                step += 1
                logits = self.PI_net(state_tensor)
                action_probs = torch.softmax(logits, dim=0)
                action_dist = torch.distributions.Categorical(action_probs)
                action_index = action_dist.sample().item()
                action = self.actions[int(action_index)]

                next_state, reward = self.env.get_next_state_and_reward(state, action)
                episode.append((state_tensor, action_index, reward))
                done = self.env.is_target(next_state)

                state = next_state
                state_tensor = self.state_to_tensor(state)

            returns = []
            G = 0
            for _, _, reward in reversed(episode):
                G = G * self.gamma + reward
                returns.append(G)
            returns.reverse()
            returns = torch.tensor(returns, device=self.device)

            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            self.optimizer.zero_grad()
            loss = torch.tensor(0.0, requires_grad=True, device=self.device)
            for (state_tensor, action_index, _), G_tensor in zip(episode, returns):
                logits = self.PI_net(state_tensor)
                action_probs = torch.softmax(logits, dim=0)
                action_dist = torch.distributions.Categorical(action_probs)
                action_tensor = torch.tensor(action_index, device=self.device)
                log_pi = action_dist.log_prob(action_tensor)
                loss = loss - log_pi * G_tensor  # monte carlo: G --> q(s.a) | 梯度上升

            loss = loss / len(episode)
            loss_record.append(loss.item())
            loss.backward()
            self.optimizer.step()

            if ep % 10 == 0:
                print(f"{sum(loss_record[-10:])/10:.4f}", end=' ')

            if ep % (num_episodes // 10) == 0:
                if params.SHOW_GRID_WORLD:
                    target_policy = {}
                    for state in self.states:
                        state_tensor = self.state_to_tensor(state)
                        logits = self.PI_net(state_tensor)
                        action_probs = torch.softmax(logits, dim=0)
                        best_action_index = torch.argmax(action_probs, dim=0).item()
                        target_policy[state] = self.actions[int(best_action_index)]
                    self.env.render(None, target_policy, self.folder_path, 
                                    title=f"episode:{ep}/{num_episodes}")
                    
        return loss_record

    def plot_loss_(self, loss_record):
        plot_loss(loss_record, self.folder_path, 
                  title="REINFORCE Loss Curve", filename="loss_curve.png")

if __name__ == "__main__":
    reinforce = REINFORCE(params.REINFORCE_SEED)
    loss_record = reinforce.solve(params.REINFORCE_EPISODE, params.REINFORCE_MAX_EPISODE_LENGTH)
    reinforce.plot_loss_(loss_record)
