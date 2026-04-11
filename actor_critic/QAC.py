from pathlib import Path
from datetime import datetime
import random

import numpy as np
import torch
import torch.nn as nn

project_root = Path(__file__).resolve().parent.parent

from src.grid_world import GridWorld
from src.plot_utils import plot_loss

class QAC:
    def __init__(
        self,
        env,
        seed,
        gamma,
        actor_lr,
        critic_lr,
        actor_net_size,
        critic_net_size,
    ) -> None:
        self.set_seed(seed)

        self.env = env
        self.start_pos = self.env.start_state
        
        self.states = self.env.states
        self.n_states = len(self.states)
        self.actions = self.env.actions
        self.n_actions = len(self.actions)

        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.device = self.get_device()
        self.actor = nn.Sequential(
            nn.Linear(2, actor_net_size),
            nn.ReLU(),
            nn.Linear(actor_net_size, self.n_actions)
        )
        self.actor.to(self.device)
        self.critic = nn.Sequential(
            nn.Linear(2, critic_net_size),
            nn.ReLU(),
            nn.Linear(critic_net_size, self.n_actions)
        )
        self.critic.to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.folder_path = str(project_root / "renders" / "QAC" / 
                               f"{self.actor_lr}_{self.critic_lr}_{self.timestamp[-6:]}")

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

    def sample_action(self, state_tensor):
        """Sample an action index from the policy network given a state tensor."""
        action_probs = torch.softmax(self.actor(state_tensor), dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)
        action_index = action_dist.sample()

        log_prob = action_dist.log_prob(action_index)
        return action_index.item(), log_prob

    def update_actor_with_normalized_advantage(self, log_probs, advantages):
        if not log_probs:
            return None

        advantages_tensor = torch.stack(advantages)
        adv_mean = advantages_tensor.mean()
        adv_std = advantages_tensor.std(unbiased=False)
        normalized_advantages = (advantages_tensor - adv_mean) / (adv_std + 1e-8)

        actor_loss = -torch.stack(log_probs) * normalized_advantages
        actor_loss = actor_loss.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss.item()

    def solve(self, n_episodes, actor_update_interval, max_episode_length):
        critic_loss_record = []
        actor_loss_record = []
        log_interval = max(1, n_episodes // 10)
        for ep in range(1, n_episodes + 1):
            state = self.start_pos
            state_tensor = self.state_to_tensor(state)
            episode_log_probs = []
            episode_advantages = []

            done = False
            for step in range(max_episode_length):
                action_index, log_prob = self.sample_action(state_tensor)
                action = self.actions[int(action_index)]
                critic_value = self.critic(state_tensor)[action_index]

                next_state, reward = self.env.get_next_state_and_reward(state, action)
                next_state_tensor = self.state_to_tensor(next_state)

                TD_target = torch.tensor(reward, device=self.device, dtype=torch.float32)
                if self.env.is_target(next_state):
                    done = True
                else:
                    with torch.no_grad():
                        next_q_values = self.critic(next_state_tensor)
                        next_action_index, _ = self.sample_action(next_state_tensor)
                        TD_target += self.gamma * next_q_values[next_action_index]
                
                advantage = TD_target - critic_value
                critic_loss = 0.5 * advantage.pow(2)

                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                # Collect data for episode-level advantage normalization.
                episode_log_probs.append(log_prob)
                episode_advantages.append(advantage.detach())

                if len(episode_log_probs) >= actor_update_interval:
                    actor_loss_value = self.update_actor_with_normalized_advantage(
                        episode_log_probs,
                        episode_advantages,
                    )
                    if actor_loss_value is not None:
                        actor_loss_record.append(actor_loss_value)
                    episode_log_probs = []
                    episode_advantages = []

                critic_loss_record.append(critic_loss.item())

                if done:
                    break
                state = next_state
                state_tensor = next_state_tensor

            # Update actor with normalized advantages over the whole episode.
            if episode_log_probs:
                actor_loss_value = self.update_actor_with_normalized_advantage(
                    episode_log_probs,
                    episode_advantages,
                )
                if actor_loss_value is not None:
                    actor_loss_record.append(actor_loss_value)
            
            if ep % log_interval == 0:
                actor_policy = {}
                critic_policy = {}
                state_values = {}
                for state in self.states:
                    state_tensor = self.state_to_tensor(state)
                    with torch.no_grad():
                        actor_action = torch.argmax(self.actor(state_tensor)).item()
                        critic_qvalues = self.critic(state_tensor)
                        critic_action = torch.argmax(critic_qvalues).item()

                    actor_policy[state] = self.actions[int(actor_action)]
                    critic_policy[state] = self.actions[int(critic_action)]
                    state_values[state] = critic_qvalues.max().item()

                self.env.render(None, actor_policy,
                                title=f"actor policy at episode {ep}",
                                folder_path=self.folder_path,
                                file_name=f"actor_policy_ep{ep}")
                self.env.render(state_values, critic_policy, title=f"critic policy at episode {ep}",
                                folder_path=self.folder_path,
                                file_name=f"critic_policy_ep{ep}")
        plot_loss(
            critic_loss_record,
            title="critic loss",
            out_dir=self.folder_path,
            file_name="critic_loss",
            x_label="Update Step",
        )
        plot_loss(
            actor_loss_record,
            title="actor loss",
            out_dir=self.folder_path,
            file_name="actor_loss",
            x_label="Update Step",
        )


if __name__ == "__main__":
    config = {
        "grid_size": 3,
        "start_pos": (0, 0),
        "target_pos": (2, 2),
        "forbidden_cells": [(0, 2), (2, 1)],
        "r_target": 1,
        "r_boundary": -1,
        "r_forbidden": -1,
        "r_step": 0,
        "r_stay": -0.1,
        "seed": 42,
        "gamma": 0.95,
        "actor_lr": 2e-3,
        "critic_lr": 8e-3,
        "actor_net_size": 16,
        "critic_net_size": 16,
        "actor_update_interval": 5,
        "max_episode_length": 30,
        "n_episodes": 10000,
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

    qac = QAC(
        env=env,
        seed=config["seed"],
        gamma=config["gamma"],
        actor_lr=config["actor_lr"],
        critic_lr=config["critic_lr"],
        actor_net_size=config["actor_net_size"],
        critic_net_size=config["critic_net_size"],
    )
    qac.solve(
        n_episodes=config["n_episodes"],
        actor_update_interval=config["actor_update_interval"],
        max_episode_length=config["max_episode_length"],
    )