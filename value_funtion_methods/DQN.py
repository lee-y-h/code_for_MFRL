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

class DQN:
    def __init__(
        self,
        env,
        seed,
        hidden_size,
        lr,
        n_replay_episodes,
        replay_episode_max_length,
        batch_size,
        n_epochs,
        discount_factor,
        target_update_freq,
        bootstrap_terminal,
    ):
        self.set_seed(seed)

        self.env = env
        self.start_pos = self.env.start_state
        self.grid_size = self.env.width
        self.n_replay_episodes = n_replay_episodes
        self.replay_episode_max_length = replay_episode_max_length
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.discount_factor = discount_factor
        self.target_update_freq = target_update_freq
        self.bootstrap_terminal = bootstrap_terminal

        self.device = self.get_device()
        self.in_dim = 2
        self.n_actions = len(self.env.actions)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.folder_path = str(project_root / "renders" / "DQN" / f"{self.timestamp}")

        self.main_net = nn.Sequential(
            nn.Linear(self.in_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.n_actions),
        ).to(self.device)
        self.target_net = nn.Sequential(
            nn.Linear(self.in_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.n_actions),
        ).to(self.device)
        self.target_net.load_state_dict(self.main_net.state_dict())

        self.optimizer = torch.optim.Adam(self.main_net.parameters(), lr=lr)

        self.behavior_probs = {
            state: {a: 1.0 / self.n_actions for a in self.env.actions}
            for state in self.env.states
        }

    def set_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    def get_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        return device

    def episodes_to_dataloader(self, episodes):
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        action_to_index = {a: i for i, a in enumerate(self.env.actions)}

        for s, a, r, s_next, done in episodes:
            states.append((s[0] / (self.grid_size - 1), s[1] / (self.grid_size - 1)))
            actions.append(action_to_index[a])
            rewards.append(r)
            next_states.append((s_next[0] / (self.grid_size - 1), s_next[1] / (self.grid_size - 1)))
            dones.append(float(done))

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        terminal_count = int((dones == 1.0).sum().item())
        print(f"{terminal_count}/{len(dones)} terminal transitions in the dataset.")

        dataset = TensorDataset(states, actions, rewards, next_states, dones)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def get_state_value_and_policy(self, net):
        state_value = {}
        policy = {}

        for state in self.env.states:
            state_tensor = torch.tensor(
                (state[0] / (self.grid_size - 1), state[1] / (self.grid_size - 1)),
                dtype=torch.float32,
            )
            q_s = net(state_tensor).squeeze(0)
            best_idx = int(q_s.argmax().item())
            q_max_val = float(q_s[best_idx].item())
            policy[state] = self.env.actions[best_idx]
            state_value[state] = q_max_val

        return state_value, policy

    def _build_replay_buffer(self):
        episodes = []
        for _ in range(self.n_replay_episodes):
            episode = self.env.generate_stochastic_episode(
                self.start_pos,
                self.behavior_probs,
                max_length=self.replay_episode_max_length,
            )
            episodes.extend(episode)

        replay_buffer = self.episodes_to_dataloader(episodes)
        print(f"length of replay_buffer: {len(replay_buffer)}")
        return replay_buffer

    def solve(self):
        replay_buffer = self._build_replay_buffer()

        loss_record = []
        render_freq = max(1, self.n_epochs // 10)

        for ep in range(1, self.n_epochs + 1):
            step = 0
            for s, a_index, r, ns, done in replay_buffer:
                step += 1

                with torch.no_grad():
                    q_t_s = self.target_net(ns)
                    q_max = q_t_s.max(dim=1).values
                    terminal_multiplier = 1.0 if self.bootstrap_terminal else (1.0 - done)
                    y_t = r + self.discount_factor * q_max * terminal_multiplier

                q_s = self.main_net(s)
                q_s_a = q_s.gather(1, a_index.unsqueeze(1)).squeeze(1)

                loss = F.mse_loss(y_t, q_s_a)
                loss_record.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if step % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.main_net.state_dict())

            if ep % render_freq == 0:
                state_value, policy = self.get_state_value_and_policy(self.target_net)
                self.env.render(
                    state_value,
                    policy,
                    self.folder_path,
                    title=f"epoch:{ep}/{self.n_epochs}",
                )

        plot_loss(
            loss_record,
            out_dir=self.folder_path,
            title="DQN Loss Curve",
            file_name="loss_curve",
            x_label="Update Step",
        )


if __name__ == "__main__":
    config = {
        "grid_size": 5,
        "start_pos": (0, 0),
        "target_pos": (2, 3),
        "forbidden_cells": [(1, 1), (1, 3), (1, 4), (2, 1), (2, 2), (3, 3)],
        "r_target": 1,
        "r_boundary": -3,
        "r_forbidden": -3,
        "r_step": -0.05,
        "r_stay": -0.1,
        "seed": 42,
        "hidden_size": 100,
        "lr": 4e-3,
        "n_replay_episodes": 200,
        "replay_episode_max_length": 1000,
        "batch_size": 32,
        "n_epochs": 200,
        "discount_factor": 0.9,
        "target_update_freq": 20,
        "bootstrap_terminal": True,
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

    dqn = DQN(
        env=env,
        seed=config["seed"],
        hidden_size=config["hidden_size"],
        lr=config["lr"],
        n_replay_episodes=config["n_replay_episodes"],
        replay_episode_max_length=config["replay_episode_max_length"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        discount_factor=config["discount_factor"],
        target_update_freq=config["target_update_freq"],
        bootstrap_terminal=config["bootstrap_terminal"],
    )
    dqn.solve()