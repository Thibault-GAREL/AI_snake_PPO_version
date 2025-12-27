"""
Script de test pour vérifier que l'implémentation PPO fonctionne correctement
"""
import torch
import numpy as np
import ia

def test_agent_creation():
    """Test la création de l'agent"""
    print("Test 1: Création de l'agent PPO...")
    agent = ia.create_agent(state_dim=16, action_dim=4)
    print(f"✓ Agent créé avec succès")
    print(f"  - Device: {agent.device}")
    print(f"  - State dim: {agent.state_dim}")
    print(f"  - Action dim: {agent.action_dim}")
    return agent

def test_action_selection(agent):
    """Test la sélection d'actions"""
    print("\nTest 2: Sélection d'actions...")

    # État fictif (16 valeurs)
    state = np.random.rand(16).tolist()

    # Test select_action (pour l'inférence)
    action = agent.select_action(state)
    print(f"✓ select_action fonctionne: action = {action}")
    assert 0 <= action < 4, "Action invalide!"

    # Test select_action_with_log_prob (pour l'entraînement)
    action, log_prob, state_value = agent.select_action_with_log_prob(state)
    print(f"✓ select_action_with_log_prob fonctionne:")
    print(f"  - Action: {action}")
    print(f"  - Log prob: {log_prob}")
    print(f"  - State value: {state_value}")

    return state, action, log_prob, state_value

def test_buffer(agent, state, action, log_prob, state_value):
    """Test le buffer de trajectoires"""
    print("\nTest 3: Buffer de trajectoires...")

    # Ajouter quelques transitions
    for i in range(10):
        reward = np.random.rand()
        done = (i == 9)
        agent.buffer.push(state, action, reward, state_value, log_prob, done)

    print(f"✓ Buffer rempli: {len(agent.buffer)} transitions")

    # Tester get()
    states, actions, rewards, values, log_probs, dones = agent.buffer.get()
    print(f"✓ Récupération des données:")
    print(f"  - States: {len(states)}")
    print(f"  - Actions: {len(actions)}")
    print(f"  - Rewards: {len(rewards)}")

def test_training(agent):
    """Test l'étape d'entraînement"""
    print("\nTest 4: Entraînement...")

    # Remplir le buffer avec des données fictives
    for episode in range(5):
        for step in range(20):
            state = np.random.rand(16).tolist()
            action, log_prob, state_value = agent.select_action_with_log_prob(state)
            reward = np.random.rand() - 0.5
            done = (step == 19)

            agent.buffer.push(state, action, reward, state_value, log_prob, done)

        # Entraîner
        agent.train_step()

    print(f"✓ Entraînement réussi (5 épisodes)")
    print(f"  - Buffer vidé: {len(agent.buffer) == 0}")

def test_save_load(agent):
    """Test la sauvegarde et le chargement"""
    print("\nTest 5: Sauvegarde et chargement...")

    import os
    os.makedirs("test_models", exist_ok=True)

    # Sauvegarder
    agent.save_model("test_models/test_model.pth")
    print(f"✓ Modèle sauvegardé")

    # Créer un nouvel agent et charger
    new_agent = ia.create_agent(state_dim=16, action_dim=4)
    new_agent.load_model("test_models/test_model.pth")
    print(f"✓ Modèle chargé")

    # Nettoyer
    os.remove("test_models/test_model.pth")
    os.rmdir("test_models")
    print(f"✓ Fichiers de test nettoyés")

def main():
    print("=" * 60)
    print("Tests de l'implémentation PPO pour Snake")
    print("=" * 60)

    try:
        # Test 1: Création
        agent = test_agent_creation()

        # Test 2: Actions
        state, action, log_prob, state_value = test_action_selection(agent)

        # Test 3: Buffer
        test_buffer(agent, state, action, log_prob, state_value)

        # Test 4: Entraînement
        test_training(agent)

        # Test 5: Sauvegarde/Chargement
        test_save_load(agent)

        print("\n" + "=" * 60)
        print("✓ TOUS LES TESTS SONT PASSÉS!")
        print("=" * 60)
        print("\nL'implémentation PPO est prête à être utilisée.")
        print("Vous pouvez maintenant lancer: python main.py")

    except Exception as e:
        print(f"\n✗ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
