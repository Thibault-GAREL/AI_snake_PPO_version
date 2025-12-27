import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import os

# Hyperparamètres PPO
nb_loop_train = 10000  # Nombre d'épisodes d'entraînement
gamma = 0.99  # Facteur de discount
lambda_gae = 0.95  # GAE lambda pour l'estimation des avantages
epsilon_clip = 0.2  # Clipping pour PPO
c1 = 0.5  # Coefficient de la value loss
c2 = 0.01  # Coefficient de l'entropy bonus
learning_rate = 3e-4
epochs_per_update = 10  # Nombre d'époques de mise à jour par batch
batch_size = 64
max_grad_norm = 0.5  # Gradient clipping


class ActorCritic(nn.Module):
    """
    Réseau Actor-Critic pour PPO
    Actor: produit une distribution de probabilités sur les actions
    Critic: estime la fonction valeur V(s)
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()

        # Couches partagées
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor head (politique)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic head (fonction valeur)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, state):
        """Forward pass pour obtenir la politique et la valeur"""
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)

        shared_features = self.shared(state)
        action_probs = self.actor(shared_features)
        state_value = self.critic(shared_features)

        return action_probs, state_value

    def act(self, state):
        """Sélectionne une action selon la politique"""
        action_probs, state_value = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)

        return action.item(), action_log_prob, state_value


class RolloutBuffer:
    """
    Buffer pour stocker les trajectoires (rollouts) pendant l'entraînement PPO
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.state_values = []
        self.log_probs = []
        self.dones = []

    def push(self, state, action, reward, state_value, log_prob, done):
        """Ajoute une transition au buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.state_values.append(state_value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def clear(self):
        """Vide le buffer"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.state_values.clear()
        self.log_probs.clear()
        self.dones.clear()

    def get(self):
        """Récupère toutes les trajectoires"""
        return (self.states, self.actions, self.rewards,
                self.state_values, self.log_probs, self.dones)

    def __len__(self):
        return len(self.states)


class PPOAgent:
    """
    Agent PPO (Proximal Policy Optimization)
    """
    def __init__(self, state_dim, action_dim, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Using device: {self.device}")

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Réseau Actor-Critic
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Buffer pour les trajectoires
        self.buffer = RolloutBuffer()

        # Pour compatibilité avec l'ancien code
        self.replay_buffer = self.buffer

    def select_action(self, state, action_dim=None):
        """
        Sélectionne une action selon la politique actuelle
        Compatible avec l'interface existante
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, state_value = self.policy.act(state_tensor)

        return action

    def select_action_with_log_prob(self, state):
        """
        Sélectionne une action et retourne aussi le log_prob et la valeur
        Utilisé pendant l'entraînement
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action, log_prob, state_value = self.policy.act(state_tensor)

        return action, log_prob, state_value

    def compute_gae(self, rewards, values, dones, next_value):
        """
        Calcul des avantages avec GAE (Generalized Advantage Estimation)
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
        Effectue une étape d'entraînement PPO
        """
        if len(self.buffer) == 0:
            return

        # Récupérer les données du buffer
        states, actions, rewards, old_values, old_log_probs, dones = self.buffer.get()

        # Calculer la valeur du dernier état pour GAE
        with torch.no_grad():
            if dones[-1]:
                next_value = 0
            else:
                next_state = torch.FloatTensor(states[-1]).unsqueeze(0).to(self.device)
                _, next_value = self.policy.forward(next_state)
                next_value = next_value.item()

        # Extraire les valeurs des tenseurs
        old_values_list = [v.item() if torch.is_tensor(v) else v for v in old_values]

        # Calculer les avantages et les returns avec GAE
        advantages, returns = self.compute_gae(rewards, old_values_list, dones, next_value)

        # Convertir en tensors
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        old_log_probs_tensor = torch.stack(old_log_probs).detach().to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)

        # Normaliser les avantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # Entraînement sur plusieurs époques
        for _ in range(epochs_per_update):
            # Forward pass
            action_probs, state_values = self.policy.forward(states_tensor)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions_tensor)
            entropy = dist.entropy()

            # Ratio pour PPO
            ratio = torch.exp(new_log_probs - old_log_probs_tensor)

            # Surrogate loss avec clipping
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1 - epsilon_clip, 1 + epsilon_clip) * advantages_tensor
            actor_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            state_values = state_values.squeeze()
            critic_loss = nn.MSELoss()(state_values, returns_tensor)

            # Entropy bonus (pour encourager l'exploration)
            entropy_loss = -entropy.mean()

            # Loss totale
            loss = actor_loss + c1 * critic_loss + c2 * entropy_loss

            # Backward pass et optimisation
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
            self.optimizer.step()

        # Vider le buffer après l'entraînement
        self.buffer.clear()

    def update_target(self):
        """
        Pour compatibilité avec l'ancien code DQN
        PPO n'utilise pas de réseau cible
        """
        pass

    def save_model(self, filepath):
        """Sauvegarde le modèle"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Charge le modèle"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {filepath}")


# Fonction pour créer l'agent (pour compatibilité)
def create_agent(state_dim=16, action_dim=4):
    """Crée et retourne un agent PPO"""
    return PPOAgent(state_dim, action_dim)
