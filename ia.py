# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import os

# PPO Hyperparameters
nb_loop_train = 10000  # Number of training episodes
gamma = 0.99  # Discount factor
lambda_gae = 0.95  # GAE lambda for advantage estimation
epsilon_clip = 0.2  # Clipping for PPO
c1 = 0.5  # Value loss coefficient
c2 = 0.01  # Entropy bonus coefficient
learning_rate = 3e-4
epochs_per_update = 10  # Number of update epochs per batch
batch_size = 64
max_grad_norm = 0.5  # Gradient clipping


class ActorCritic(nn.Module):
    """
    Actor-Critic Network for PPO
    Actor: produces action probability distribution
    Critic: estimates value function V(s)
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, state):
        """Forward pass to get policy and value"""
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)

        shared_features = self.shared(state)
        action_probs = self.actor(shared_features)
        state_value = self.critic(shared_features)

        return action_probs, state_value

    def act(self, state):
        """Select an action according to the policy"""
        action_probs, state_value = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)

        return action.item(), action_log_prob, state_value


class RolloutBuffer:
    """
    Buffer for storing trajectories (rollouts) during PPO training
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.state_values = []
        self.log_probs = []
        self.dones = []

    def push(self, state, action, reward, state_value, log_prob, done):
        """Add a transition to the buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.state_values.append(state_value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def clear(self):
        """Clear the buffer"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.state_values.clear()
        self.log_probs.clear()
        self.dones.clear()

    def get(self):
        """Retrieve all trajectories"""
        return (self.states, self.actions, self.rewards,
                self.state_values, self.log_probs, self.dones)

    def __len__(self):
        return len(self.states)


class PPOAgent:
    """
    PPO Agent (Proximal Policy Optimization)
    """
    def __init__(self, state_dim, action_dim, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Using device: {self.device}")

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Actor-Critic Network
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Buffer for trajectories
        self.buffer = RolloutBuffer()

        # For compatibility with old code
        self.replay_buffer = self.buffer

    def select_action(self, state, action_dim=None):
        """
        Select an action according to current policy
        Compatible with existing interface
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, state_value = self.policy.act(state_tensor)

        return action

    def select_action_with_log_prob(self, state):
        """
        Select an action and also return log_prob and value
        Used during training
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action, log_prob, state_value = self.policy.act(state_tensor)

        return action, log_prob, state_value

    def compute_gae(self, rewards, values, dones, next_value):
        """
        Compute advantages using GAE (Generalized Advantage Estimation)
        """
        advantages = []
        gae = 0

        values = values + [next_value]

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = values[t + 1]

            delta = rewards[t] + gamma * next_value_t * next_non_terminal - values[t]
            gae = delta + gamma * lambda_gae * next_non_terminal * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, values[:-1])]

        return advantages, returns

    def train_step(self, batch_size=None):
        """
        Perform a PPO training step
        """
        if len(self.buffer) == 0:
            return

        # Get data from buffer
        states, actions, rewards, old_values, old_log_probs, dones = self.buffer.get()

        # Calculate value of last state for GAE
        with torch.no_grad():
            if dones[-1]:
                next_value = 0
            else:
                next_state = torch.FloatTensor(states[-1]).unsqueeze(0).to(self.device)
                _, next_value = self.policy.forward(next_state)
                next_value = next_value.item()

        # Extract values from tensors
        old_values_list = [v.item() if torch.is_tensor(v) else v for v in old_values]

        # Compute advantages and returns with GAE
        advantages, returns = self.compute_gae(rewards, old_values_list, dones, next_value)

        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        old_log_probs_tensor = torch.stack(old_log_probs).detach().to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # Training over multiple epochs
        for _ in range(epochs_per_update):
            # Forward pass
            action_probs, state_values = self.policy.forward(states_tensor)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions_tensor)
            entropy = dist.entropy()

            # Ratio for PPO
            ratio = torch.exp(new_log_probs - old_log_probs_tensor)

            # Surrogate loss with clipping
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1 - epsilon_clip, 1 + epsilon_clip) * advantages_tensor
            actor_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            state_values = state_values.squeeze()
            critic_loss = nn.MSELoss()(state_values, returns_tensor)

            # Entropy bonus (for exploration)
            entropy_loss = -entropy.mean()

            # Total loss
            loss = actor_loss + c1 * critic_loss + c2 * entropy_loss

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
            self.optimizer.step()

        # Clear buffer after training
        self.buffer.clear()

    def update_target(self):
        """
        For compatibility with old DQN code
        PPO does not use target network
        """
        pass

    def save_model(self, filepath):
        """Save the model"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load the model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {filepath}")


# Function to create agent (for compatibility)
def create_agent(state_dim=16, action_dim=4):
    """Create and return a PPO agent"""
    return PPOAgent(state_dim, action_dim)
