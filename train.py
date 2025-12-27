"""
Script d'entraînement optimisé pour l'IA Snake avec PPO
"""
import os
import sys
import progressbar
import time

import snake
import ia

def train(nb_episodes=10000, model_name="models_ppo", load_existing=True, save_interval=100):
    """
    Entraîne l'agent PPO sur Snake

    Args:
        nb_episodes: Nombre d'épisodes d'entraînement
        model_name: Nom du dossier où sauvegarder le modèle
        load_existing: Charger un modèle existant si disponible
        save_interval: Intervalle de sauvegarde (en épisodes)
    """
    print("=" * 70)
    print("Entraînement de l'IA Snake avec PPO")
    print("=" * 70)

    # Créer l'agent PPO
    state_dim = 16  # 8 distances aux obstacles + 8 distances à la nourriture
    action_dim = 4  # UP, RIGHT, DOWN, LEFT
    agent = ia.create_agent(state_dim, action_dim)

    print(f"\nConfiguration:")
    print(f"  - Épisodes: {nb_episodes}")
    print(f"  - Save interval: {save_interval}")
    print(f"  - Model name: {model_name}")
    print(f"  - Device: {agent.device}")
    print(f"  - Learning rate: {ia.learning_rate}")
    print(f"  - Gamma: {ia.gamma}")
    print(f"  - Epsilon clip: {ia.epsilon_clip}")

    # Charger le modèle existant si disponible
    model_path = f"{model_name}/snake_ppo_model.pth"
    if load_existing and os.path.exists(model_path):
        agent.load_model(model_path)
        print(f"\n✓ Modèle chargé: {model_path}")
    else:
        os.makedirs(model_name, exist_ok=True)
        print(f"\n✓ Nouveau modèle créé")

    # Variables pour le tracking
    scores = []
    score_temp = 0
    best_score = 0
    best_avg_score = 0

    print(f"\n{'='*70}")
    print("Début de l'entraînement...")
    print(f"{'='*70}\n")

    # Boucle d'entraînement
    start_time = time.time()

    for episode in progressbar.progressbar(range(nb_episodes)):
        # Jouer un épisode
        score = snake.game_loop(snake.rect_width, snake.rect_height, snake.display, agent)
        score_temp += score
        scores.append(score)

        # Mettre à jour le meilleur score
        if score > best_score:
            best_score = score

        # Entraîner l'agent
        agent.train_step(batch_size=64)

        # Sauvegarder et afficher les statistiques
        if (episode + 1) % save_interval == 0:
            avg_score = score_temp / save_interval

            # Mettre à jour le meilleur score moyen
            if avg_score > best_avg_score:
                best_avg_score = avg_score

            # Calculer le temps écoulé
            elapsed_time = time.time() - start_time
            eps_per_sec = (episode + 1) / elapsed_time

            print(f"\n{'='*70}")
            print(f"Épisode {episode + 1}/{nb_episodes}")
            print(f"  - Score moyen (last {save_interval}): {avg_score:.2f}")
            print(f"  - Meilleur score: {best_score}")
            print(f"  - Meilleur score moyen: {best_avg_score:.2f}")
            print(f"  - Vitesse: {eps_per_sec:.2f} eps/s")
            print(f"  - Temps écoulé: {elapsed_time/60:.1f} min")
            print(f"{'='*70}")

            # Sauvegarder le modèle
            agent.save_model(model_path)

            # Sauvegarder aussi le meilleur modèle
            if avg_score >= best_avg_score:
                best_model_path = f"{model_name}/snake_ppo_model_best.pth"
                agent.save_model(best_model_path)
                print(f"✓ Nouveau meilleur modèle sauvegardé!")

            score_temp = 0

            # Afficher quelques statistiques supplémentaires
            if len(scores) >= 100:
                recent_avg = sum(scores[-100:]) / 100
                print(f"  - Score moyen (last 100 eps): {recent_avg:.2f}")

    # Fin de l'entraînement
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print("Entraînement terminé!")
    print(f"{'='*70}")
    print(f"Temps total: {total_time/60:.1f} minutes")
    print(f"Meilleur score: {best_score}")
    print(f"Meilleur score moyen: {best_avg_score:.2f}")
    print(f"Modèle final sauvegardé: {model_path}")
    print(f"{'='*70}\n")

    return agent, scores


if __name__ == "__main__":
    # Configuration
    NB_EPISODES = 10000
    MODEL_NAME = "models_ppo"
    LOAD_EXISTING = True
    SAVE_INTERVAL = 100

    # Lancer l'entraînement
    agent, scores = train(
        nb_episodes=NB_EPISODES,
        model_name=MODEL_NAME,
        load_existing=LOAD_EXISTING,
        save_interval=SAVE_INTERVAL
    )

    print("Vous pouvez maintenant tester votre modèle avec:")
    print("  python test_model.py")
