"""
Script pour tester et visualiser l'IA Snake entraînée
"""
import sys
import os

# Activer l'affichage
import snake
snake.show = True
snake.player = False
snake.stop_iteration = 1000  # Augmenter pour des parties plus longues

import ia

def play_game(model_path, nb_games=5):
    """
    Fait jouer l'IA entraînée

    Args:
        model_path: Chemin vers le modèle entraîné
        nb_games: Nombre de parties à jouer
    """
    print("=" * 70)
    print("Test de l'IA Snake entraînée")
    print("=" * 70)

    # Créer l'agent
    state_dim = 16
    action_dim = 4
    agent = ia.create_agent(state_dim, action_dim)

    # Charger le modèle
    if os.path.exists(model_path):
        agent.load_model(model_path)
        print(f"✓ Modèle chargé: {model_path}\n")
    else:
        print(f"✗ Erreur: Modèle non trouvé: {model_path}")
        print(f"Assurez-vous d'avoir entraîné le modèle d'abord avec:")
        print(f"  python train.py")
        return

    # Réinitialiser pygame avec affichage
    import pygame
    pygame.init()
    snake.display = pygame.display.set_mode((snake.width, snake.height))
    pygame.display.set_caption("Snake - IA PPO")
    snake.clock = pygame.time.Clock()
    snake.fonttype = pygame.font.SysFont(None, 30)

    scores = []

    print(f"Lancement de {nb_games} partie(s)...")
    print(f"Appuyez sur ESC pour quitter\n")
    print("=" * 70)

    # Jouer plusieurs parties
    for game_num in range(nb_games):
        print(f"\nPartie {game_num + 1}/{nb_games}")
        score = snake.game_loop(snake.rect_width, snake.rect_height, snake.display, agent)
        scores.append(score)
        print(f"  Score: {score}")

        # Pause entre les parties
        if game_num < nb_games - 1:
            print("  Prochaine partie dans 2 secondes...")
            pygame.time.wait(2000)

    # Statistiques finales
    print("\n" + "=" * 70)
    print("Statistiques finales:")
    print(f"  - Nombre de parties: {len(scores)}")
    print(f"  - Score total: {sum(scores)}")
    print(f"  - Score moyen: {sum(scores)/len(scores):.2f}")
    print(f"  - Meilleur score: {max(scores)}")
    print(f"  - Pire score: {min(scores)}")
    print("=" * 70)

    pygame.quit()


if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "models_ppo/snake_ppo_model.pth"
    NB_GAMES = 5

    # Vous pouvez tester le meilleur modèle en décommentant:
    # MODEL_PATH = "models_ppo/snake_ppo_model_best.pth"

    # Ou passer le chemin en argument
    if len(sys.argv) > 1:
        MODEL_PATH = sys.argv[1]

    if len(sys.argv) > 2:
        NB_GAMES = int(sys.argv[2])

    # Lancer le test
    play_game(MODEL_PATH, NB_GAMES)
