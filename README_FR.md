# AI Snake avec PPO (Proximal Policy Optimization)

Une IA qui apprend à jouer au jeu Snake en utilisant l'algorithme PPO (Proximal Policy Optimization).

## 🎮 Présentation

Ce projet implémente un agent d'apprentissage par renforcement qui apprend à jouer au jeu Snake. L'algorithme utilisé est **PPO** (Proximal Policy Optimization), une méthode state-of-the-art pour l'apprentissage par renforcement.

### Caractéristiques

- ✅ **Algorithme PPO complet** avec Actor-Critic
- ✅ **Support GPU** pour accélérer l'entraînement (CUDA)
- ✅ **GAE** (Generalized Advantage Estimation) pour de meilleures estimations
- ✅ **Sauvegarde/Chargement** automatique des modèles
- ✅ **Visualisation** de l'entraînement avec graphiques Excel
- ✅ **Mode visible/invisible** pour accélérer l'entraînement

## 📋 Prérequis

- Python 3.8+
- CUDA (optionnel, pour utiliser le GPU)
- Carte graphique NVIDIA (optionnel, pour l'accélération)

## 🚀 Installation

1. **Cloner le repository** (ou télécharger les fichiers)

2. **Installer les dépendances** :
```bash
pip install -r requirements.txt
```

3. **Vérifier l'installation de PyTorch avec CUDA** (optionnel) :
```bash
python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}')"
```

## 🎯 Utilisation

### Entraîner l'IA

Pour lancer l'entraînement :

```bash
python main.py
```

L'entraînement va :
- Créer un agent PPO
- Entraîner pendant 10 000 épisodes (configurable dans `ia.py`)
- Sauvegarder le modèle automatiquement tous les 100 épisodes
- Créer un fichier Excel avec les statistiques d'entraînement

### Configuration

Vous pouvez modifier les paramètres dans `snake.py` et `ia.py` :

#### Dans `snake.py` :
```python
show = False  # True pour voir le jeu, False pour entraîner plus vite
player = False  # True pour jouer manuellement
stop_iteration = 100  # Nombre max de mouvements par partie
```

#### Dans `ia.py` :
```python
nb_loop_train = 10000  # Nombre d'épisodes d'entraînement
gamma = 0.99  # Facteur de discount
epsilon_clip = 0.2  # Clipping PPO
learning_rate = 3e-4  # Taux d'apprentissage
```

### Regarder l'IA jouer

Une fois l'entraînement terminé, vous pouvez regarder l'IA jouer :

1. Dans `snake.py`, changez :
```python
show = True  # Activer l'affichage
```

2. Créez un script de test (par exemple `test.py`) :
```python
import snake
import ia

# Charger le modèle entraîné
agent = ia.create_agent(16, 4)
agent.load_model("models_ppo1/snake_ppo_model.pth")

# Jouer une partie
score = snake.game_loop(snake.rect_width, snake.rect_height, snake.display, agent)
print(f"Score final: {score}")
```

## 🧠 Architecture de l'IA

### État (Observations)
L'agent observe 16 valeurs :
- 8 distances aux obstacles (murs ou corps du serpent) dans 8 directions
- 8 distances à la nourriture dans 8 directions

### Actions
4 actions possibles :
- 0 : Haut (UP)
- 1 : Droite (RIGHT)
- 2 : Bas (DOWN)
- 3 : Gauche (LEFT)

### Récompenses
- **+1** : Manger la nourriture
- **-1** : Mourir (collision avec mur ou soi-même)

### Réseau de neurones

**Architecture Actor-Critic** :
```
Input (16)
    ↓
Shared layers: Linear(16→256) → ReLU → Linear(256→256) → ReLU
    ↓
    ├→ Actor: Linear(256→128) → ReLU → Linear(128→4) → Softmax
    └→ Critic: Linear(256→128) → ReLU → Linear(128→1)
```

## 📊 Résultats

Les résultats de l'entraînement sont sauvegardés dans :
- **Modèles** : `models_ppoX/snake_ppo_model.pth`
- **Graphiques** : `donnees2.xlsx` avec un graphique d'évolution du score

## 🔧 Fichiers du projet

- `ia.py` : Implémentation de l'agent PPO
- `snake.py` : Jeu Snake et environnement
- `main.py` : Script d'entraînement principal
- `exw.py` : Utilitaires pour Excel
- `compteur.py` : Compteur d'exécutions
- `requirements.txt` : Dépendances Python

## 💡 Conseils d'optimisation

1. **Pour un entraînement plus rapide** :
   - Mettez `show = False` dans `snake.py`
   - Augmentez `stop_iteration` pour des parties plus longues
   - Utilisez un GPU (CUDA)

2. **Pour améliorer les performances** :
   - Ajustez `learning_rate` (essayez 1e-4 ou 5e-4)
   - Modifiez `gamma` (essayez 0.95 ou 0.99)
   - Augmentez `nb_loop_train` pour plus d'entraînement

3. **Si l'IA n'apprend pas** :
   - Vérifiez que le GPU est bien utilisé
   - Réduisez `epsilon_clip` (essayez 0.1)
   - Augmentez `c2` (entropy bonus) pour plus d'exploration

## 📝 Algorithme PPO

PPO (Proximal Policy Optimization) est un algorithme d'apprentissage par renforcement qui :
1. Collecte des trajectoires en jouant avec la politique actuelle
2. Calcule les avantages avec GAE
3. Met à jour la politique avec un objectif clippé pour éviter les mises à jour trop importantes
4. Met à jour le critique pour mieux estimer les valeurs

## 🤝 Contribution

N'hésitez pas à modifier et améliorer le code !

## 📄 Licence

Ce projet est libre d'utilisation.
