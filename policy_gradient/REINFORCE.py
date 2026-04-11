from pathlib import Path
from datetime import datetime
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

project_root = Path(__file__).resolve().parent.parent

from src.grid_world import GridWorld
from src.plot_utils import plot_loss

class REINFORCE:
    def __init__(self, env, seed, gamma, lr, net_size) -> None:
        self.set_seed(seed)

        self.env = env
        self.start_pos = self.env.start_state
        
        self.states = self.env.states
        self.n_states = len(self.states)
        self.actions = self.env.actions
        self.n_actions = len(self.actions)

        self.gamma = gamma
        self.lr = lr

        self.device = self.get_device()
        self.PI_net = nn.Sequential(
            nn.Linear(2, net_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(net_size, self.n_actions)
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

    def solve(self, n_episodes, episode_max_len):
        loss_record = []
        for ep in range(1, n_episodes + 1):
            state = self.start_pos
            state_tensor = self.state_to_tensor(state)

            episode = []
            done = False
            step = 0
            while not done and step < episode_max_len:
                step += 1
                logits = self.PI_net(state_tensor)
                action_probs = torch.softmax(logits, dim=-1)
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

            if ep % (n_episodes // 10) == 0:
                target_policy = {}
                for state in self.states:
                    state_tensor = self.state_to_tensor(state)
                    logits = self.PI_net(state_tensor)
                    action_probs = torch.softmax(logits, dim=0)
                    best_action_index = torch.argmax(action_probs, dim=0).item()
                    target_policy[state] = self.actions[int(best_action_index)]
                self.env.render(None, target_policy, self.folder_path, 
                                title=f"episode:{ep}/{n_episodes}")
                    
        return loss_record

    def plot_loss_(self, loss_record):
        plot_loss(
            loss_record,
            self.folder_path,
            title="REINFORCE Loss Curve",
            file_name="loss_curve.png",
            x_label="Episode",
        )

if __name__ == "__main__":
    config = {
        "grid_size": 5,
        "start_pos": (0, 0),
        "target_pos": (2, 3),
        "forbidden_cells": [(1, 1), (1, 3), (2, 1), (2, 2), (3, 3)],
        "r_target": 1,
        "r_boundary": -1,
        "r_forbidden": -1,
        "r_step": 0,
        "r_stay": 0,
        "seed": 42,
        "gamma": 0.95,
        "lr": 2e-4,
        "net_size": 128,
        "n_episodes": 50000,
        "max_episode_len": 100,
    }

    env = GridWorld(
        width=config["grid_size"],
        height=config["grid_size"],
        target=config["target_pos"],
        forbidden=config["forbidden_cells"],
        start=config["start_pos"],
        r_target=config["r_target"],
        r_boundary=config["r_boundary"],
        r_forbidden=config["r_forbidden"],
        r_step=config["r_step"],
        r_stay=config["r_stay"],
    )

    reinforce = REINFORCE(
        env=env,
        seed=config["seed"],
        gamma=config["gamma"],
        lr=config["lr"],
        net_size=config["net_size"],
    )
    loss_record = reinforce.solve(config["n_episodes"], config["max_episode_len"])
    reinforce.plot_loss_(loss_record)
