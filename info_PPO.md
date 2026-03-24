# Informations Détaillées du Modèle PPO V2

## 📊 Performances

| Métrique | Valeur |
|----------|--------|
| **Score Maximum** | 21 (approx. 21 segments) |
| **Score Moyen Meilleur** | 10.18 (fenêtre 50 épisodes) |
| **Nombre d'Épisodes** | 49,328 |
| **Nombre de Timesteps** | 5,000,080 |

## ⏱️ Temps d'Entraînement

| Paramètre | Valeur |
|-----------|--------|
| **Temps Total** | 22,413 secondes |
| **Temps en Heures** | ~6.23 heures |
| **Temps en Minutes** | ~373.55 minutes |

## 🧠 Architecture Réseau

### Entrée (État)
- **Dimension d'entrée** : 22 features
  - [0:8] Distances aux dangers (N, NE, E, SE, S, SW, W, NW) normalisées
  - [8:16] Distances nourriture (même 8 directions) normalisées
  - [16:20] One-hot encoding direction courante (UP, RIGHT, DOWN, LEFT)
  - [20] Longueur serpent normalisée (0→1)
  - [21] Urgence nourriture (steps_since_food / MAX_STEPS)

### Réseau Acteur (Policy Network)
```
Input (22) 
  ↓ Linear + ReLU
Hidden (256)
  ↓ Linear + ReLU
Hidden (256)
  ↓ Linear + Tanh
Hidden (128)
  ↓ Linear
Output Actions (4)  [UP, RIGHT, DOWN, LEFT]
```

### Réseau Critique (Value Network)
```
Hidden (256)  [partageable avec acteur]
  ↓ Linear + Tanh
Hidden (128)
  ↓ Linear
Output Value (1)
```

### Statistiques Réseau
| Paramètre | Valeur |
|-----------|--------|
| **Couche 1** | Linear(22 → 256) |
| **Couche 2** | Linear(256 → 256) |
| **Couche 3 (Actor)** | Linear(256 → 128) |
| **Couche 4 (Actor)** | Linear(128 → 4) |
| **Couche 3 (Critic)** | Linear(256 → 128) |
| **Couche 4 (Critic)** | Linear(128 → 1) |
| **Nombre de Neurones (Hidden)** | 256 |

## 💾 Configuration Mémoire & Batch

| Paramètre | Valeur |
|-----------|--------|
| **Batch Size** | 64 |
| **N_STEPS (Rollout Buffer)** | 2,048 |
| **Memory Buffer Total** | 2,048 timesteps par collect |

### Calculs
- **Nombre de mini-batches par epoch** : 2,048 ÷ 64 = 32 batches
- **Nombre total de passes training** : N_EPOCHS × 32 = 8 × 32 = 256 passes

## 🎓 Hyperparamètres d'Entraînement

### Learning Rate
| Paramètre | Valeur |
|-----------|--------|
| **Learning Rate Initial** | 3×10⁻⁴ (0.0003) |
| **Scheduler** | CosineAnnealingLR |
| **LR Minimum** | 1×10⁻⁵ |
| **Décroissance** | Cosinus sur l'intégralité de l'entraînement |

### PPO Spécifiques
| Paramètre | Valeur |
|-----------|--------|
| **Epsilon (Clip)** | 0.15 |
| **Nombre d'Epochs par Update** | 8 |
| **Coefficient Entropy** | 0.05 |
| **Coefficient Value Function** | 0.5 |
| **Gradient Max Norm** | 0.5 |

### GAE (Generalized Advantage Estimation)
| Paramètre | Valeur |
|-----------|--------|
| **Gamma (Discount)** | 0.99 |
| **Lambda (GAE)** | 0.95 |

## 🔧 Configuration Environnement

### Jeu Snake
| Paramètre | Valeur |
|-----------|--------|
| **Taille Grille** | 800×400 pixels |
| **Taille Cellule** | 50 pixels |
| **Colonnes** | 16 |
| **Lignes** | 8 |
| **Max Longueur Théorique** | 128 (16×8) |
| **Max Steps par Épisode** | 500 |

### Actions
| Paramètre | Valeur |
|-----------|--------|
| **Nombre d'Actions** | 4 |
| **Actions** | UP (0), RIGHT (1), DOWN (2), LEFT (3) |

### Récompenses
| Événement | Récompense |
|-----------|-----------|
| **Survie (par step)** | +0.02 |
| **Nourriture mangée** | +10.0 |
| **Niveau complété** | +20.0 |
| **Mort** | -10.0 - (longueur × 0.5) |
| **Tourner en rond** | -0.5 (pénalité additionnelle) |

## 📈 Optimiseur

| Paramètre | Valeur |
|-----------|--------|
| **Optimiseur** | Adam |
| **Epsilon (eps)** | 1×10⁻⁵ |

## ✅ Résumé Exécution

| Métrique | Valeur |
|----------|--------|
| **GPU/CPU** | Auto-détection (CUDA si disponible) |
| **Version** | PPO v3 |
| **Config Version** | v3 (avec reward shaping corrigé et state enrichi) |
| **Date Type Analysis** | float32 pour observations, long pour actions |

### Corrections Apportées en V3
1. **LR Schedule Corrigé** : CosineAnnealingLR au lieu de LinearLR
2. **Reward Shaping** : Suppression du shaping Manhattan, ajout de récompense survie + bonus nourriture + pénalité mort
3. **State Enrichi** : 22 features (+ longueur normalisée + urgence nourriture)

---

*Informations extraites du code : main.py, PPO.py et training_log.csv*
*Date de génération : 2026-03-24*
