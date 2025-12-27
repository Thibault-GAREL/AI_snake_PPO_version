# Guide de démarrage rapide - Snake PPO

## Installation rapide

```bash
# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation

### 1. Entraîner l'IA (recommandé)

```bash
python train.py
```

Ou utilisez le script original:
```bash
python main.py
```

L'entraînement va:
- Entraîner pendant 10 000 épisodes (~2-3 heures selon votre GPU)
- Sauvegarder automatiquement tous les 100 épisodes
- Créer un fichier Excel avec les statistiques

**Conseil**: Pour un entraînement plus rapide, assurez-vous que `show = False` dans `snake.py` (ligne 7)

### 2. Tester le modèle entraîné

```bash
python test_model.py
```

Ou pour spécifier un modèle:
```bash
python test_model.py models_ppo/snake_ppo_model_best.pth 10
```

### 3. Tester l'implémentation

```bash
python test_ppo.py
```

## Configuration rapide

### Dans `snake.py`:
```python
show = False  # True pour voir le jeu, False pour entraîner vite
stop_iteration = 100  # Mouvements max par partie
```

### Dans `ia.py`:
```python
nb_loop_train = 10000  # Nombre d'épisodes
learning_rate = 3e-4  # Taux d'apprentissage
gamma = 0.99  # Discount factor
```

## Fichiers importants

- `ia.py` - Implémentation PPO complète
- `snake.py` - Jeu Snake
- `train.py` - Script d'entraînement optimisé
- `test_model.py` - Tester le modèle entraîné
- `main.py` - Script d'entraînement original

## Résultats attendus

Après quelques milliers d'épisodes, l'IA devrait:
- Atteindre des scores de 5-15 régulièrement
- Apprendre à éviter les murs
- Suivre la nourriture efficacement

Pour de meilleurs résultats:
- Entraînez plus longtemps (20 000+ épisodes)
- Ajustez le learning rate
- Augmentez `stop_iteration` pour des parties plus longues

## Support GPU

Le code détecte automatiquement CUDA. Vérifiez avec:
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## Problèmes courants

**L'IA n'apprend pas**:
- Vérifiez que l'entraînement dure assez longtemps
- Essayez de réduire le learning rate à 1e-4
- Assurez-vous que le GPU est utilisé

**Entraînement trop lent**:
- Mettez `show = False` dans snake.py
- Utilisez un GPU
- Réduisez `stop_iteration`

Bon entraînement ! 🐍🎮🤖
