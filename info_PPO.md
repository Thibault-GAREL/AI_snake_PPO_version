# Informations Détaillées du Modèle PPO V4

## 🔍 Résumé XAI — Interprétabilité du Modèle

Quatre analyses XAI ont été menées sur le modèle PPO V2 entraîné afin d'en comprendre les mécanismes internes. L'analyse des **activations internes** (t-SNE, UMAP, spécialisation neuronale) révèle que le réseau développe des représentations structurées et séparables par situation de jeu, avec une saturation quasi-totale de la couche critique (100%) indiquant une forte confiance dans l'estimation de valeur. L'analyse des **features** (permutation importance, corrélations, poids) confirme que les distances aux dangers sont les variables les plus critiques pour la survie (première cause de chute de score en permutation), suivies de près par les distances à la nourriture qui guident la navigation directionnelle. L'analyse des **heatmaps de politique** montre que l'agent adopte une stratégie spatialement cohérente — la valeur d'état V(s) monte à l'approche de la nourriture et s'effondre à la mort — avec des zones de forte hésitation (faible gap P_max − P_2nd) uniquement dans les configurations ambiguës. Enfin, l'analyse **SHAP** (beeswarm, waterfall, heatmap globale) quantifie précisément que les features `Food_NE`, `Food_E` et `Food_SE` dominent l'influence sur les logits toutes actions confondues, et que l'agent réagit de façon logique et interprétable face aux sept situations-clés testées (danger directionnel, nourriture alignée, situation neutre).

---



## 📊 Performances

### V3 (référence — 22 features, 5M timesteps)

| Métrique                 | Valeur                      |
| ------------------------ | --------------------------- |
| **Score Maximum**        | 21 (approx. 21 segments)    |
| **Score Moyen Meilleur** | 10.18 (fenêtre 50 épisodes) |
| **Nombre d'Épisodes**    | 49,328                      |
| **Nombre de Timesteps**  | 5,000,080                   |

### V4 (26 features, 8M timesteps — à entraîner)

| Métrique                 | Valeur         |
| ------------------------ | -------------- |
| **Score Maximum**        | En attente     |
| **Score Moyen Meilleur** | En attente     |
| **Nombre de Timesteps**  | 8,000,000      |

## ⏱️ Temps d'Entraînement

### V3

| Paramètre            | Valeur          |
| -------------------- | --------------- |
| **Temps Total**      | 22,413 secondes |
| **Temps en Heures**  | ~6.23 heures    |
| **Temps en Minutes** | ~373.55 minutes |

### V4 (estimé)

| Paramètre            | Valeur           |
| -------------------- | ---------------- |
| **Temps Estimé**     | ~10 heures (GPU) |

## 🧠 Architecture Réseau

### Entrée (État)

- **Dimension d'entrée** : 28 features (unified — voir `input.md`)
  - [0:8]   Distances aux dangers (N, NE, E, SE, S, SW, W, NW) normalisées
  - [8:16]  Distances nourriture (même 8 directions) normalisées — sparses
  - [16]    `food_delta_x` : (food.x - head.x) / WIDTH — direction continue
  - [17]    `food_delta_y` : (food.y - head.y) / HEIGHT — direction continue
  - [18:22] `danger_N/E/S/W` : danger binaire immédiat, absolu (4 directions cardinales)
  - [22:26] One-hot encoding direction courante (UP, RIGHT, DOWN, LEFT)
  - [26]    Longueur serpent normalisée (0→1)
  - [27]    Urgence nourriture (steps_since_food / MAX_STEPS)

### Réseau Acteur (Policy Network)

```
Input (28)
  ↓ Linear + LayerNorm + Tanh
Hidden (256)
  ↓ Linear + LayerNorm + Tanh
Hidden (256)
  ↓ Linear + Tanh
Hidden (128)
  ↓ Linear
Output Actions (4)  [UP, RIGHT, DOWN, LEFT]
```

### Réseau Critique (Value Network)

```
Hidden (256)  [tronc partagé avec acteur]
  ↓ Linear + Tanh
Hidden (128)
  ↓ Linear
Output Value (1)
```

### Statistiques Réseau

| Paramètre                       | Valeur            |
| ------------------------------- | ----------------- |
| **Couche 1**                    | Linear(28 → 256)  |
| **Couche 2**                    | Linear(256 → 256) |
| **Couche 3 (Actor)**            | Linear(256 → 128) |
| **Couche 4 (Actor)**            | Linear(128 → 4)   |
| **Couche 3 (Critic)**           | Linear(256 → 128) |
| **Couche 4 (Critic)**           | Linear(128 → 1)   |
| **Nombre de Neurones (Hidden)** | 256               |

## 💾 Configuration Mémoire & Batch

| Paramètre                    | Valeur                      |
| ---------------------------- | --------------------------- |
| **Batch Size**               | 64                          |
| **N_STEPS (Rollout Buffer)** | 2,048                       |
| **Memory Buffer Total**      | 2,048 timesteps par collect |

### Calculs

- **Nombre de mini-batches par epoch** : 2,048 ÷ 64 = 32 batches
- **Nombre total de passes training** : N_EPOCHS × 32 = 8 × 32 = 256 passes par update

### Reward Shaping

| Evenement              | Recompense                                                          |
| ---------------------- | ------------------------------------------------------------------- |
| **Survie (par step)**  | +0.02                                                               |
| **Proximite food**     | +0.1 x (prev_manhattan - new_manhattan) / CELL *(potential-based)* |
| **Nourriture mangee**  | +10.0                                                               |
| **Niveau complete**    | +20.0                                                               |
| **Mort**               | -10.0 - (longueur x 0.5)                                           |
| **Tourner en rond**    | -0.5 (penalite additionnelle)                                       |

## 🎓 Hyperparamètres d'Entraînement

### Learning Rate

| Paramètre                 | Valeur                                      |
| ------------------------- | ------------------------------------------- |
| **Learning Rate Initial** | 3×10⁻⁴ (0.0003)                             |
| **Scheduler**             | CosineAnnealingLR                           |
| **LR Minimum**            | 1×10⁻⁵                                      |
| **Décroissance**          | Cosinus sur l'intégralité de l'entraînement |

### PPO Spécifiques

| Paramètre                      | Valeur |
| ------------------------------ | ------ |
| **Epsilon (Clip)**             | 0.15   |
| **Nombre d'Epochs par Update** | 8      |
| **Coefficient Entropy**        | 0.05   |
| **Coefficient Value Function** | 0.5    |
| **Gradient Max Norm**          | 0.5    |

### GAE (Generalized Advantage Estimation)

| Paramètre            | Valeur |
| -------------------- | ------ |
| **Gamma (Discount)** | 0.99   |
| **Lambda (GAE)**     | 0.95   |

## 🔧 Configuration Environnement

### Jeu Snake

| Paramètre                  | Valeur         |
| -------------------------- | -------------- |
| **Taille Grille**          | 800×400 pixels |
| **Taille Cellule**         | 50 pixels      |
| **Colonnes**               | 16             |
| **Lignes**                 | 8              |
| **Max Longueur Théorique** | 128 (16×8)     |
| **Max Steps par Épisode**  | 500            |

### Actions

| Paramètre            | Valeur                                |
| -------------------- | ------------------------------------- |
| **Nombre d'Actions** | 4                                     |
| **Actions**          | UP (0), RIGHT (1), DOWN (2), LEFT (3) |

### Récompenses

| Événement             | Récompense                    |
| --------------------- | ----------------------------- |
| **Survie (par step)** | +0.02                         |
| **Nourriture mangée** | +10.0                         |
| **Niveau complété**   | +20.0                         |
| **Mort**              | -10.0 - (longueur × 0.5)      |
| **Tourner en rond**   | -0.5 (pénalité additionnelle) |

## 📈 Optimiseur

| Paramètre         | Valeur |
| ----------------- | ------ |
| **Optimiseur**    | Adam   |
| **Epsilon (eps)** | 1×10⁻⁵ |

## ✅ Résumé Exécution

| Métrique               | Valeur                                            |
| ---------------------- | ------------------------------------------------- |
| **GPU/CPU**            | Auto-détection (CUDA si disponible)                |
| **Version**            | PPO v4                                             |
| **Config Version**     | v4 (26 features + potential-based reward)          |
| **Date Type Analysis** | float32 pour observations, long pour actions       |

### Corrections Apportées en V3

1. **LR Schedule Corrigé** : CosineAnnealingLR au lieu de LinearLR
2. **Reward Shaping** : Suppression du shaping Manhattan, ajout de récompense survie + bonus nourriture + pénalité mort
3. **State Enrichi** : 22 features (+ longueur normalisée + urgence nourriture)

### Améliorations Apportées en V4

1. **Unified 28-feature state** (voir `input.md`) :
   - `food_delta_x` / `food_delta_y` : direction continue vers la nourriture (les features sparses [8:16] sont quasi-nulles en dehors des alignements exacts)
   - `danger_N/E/S/W` : 4 signaux binaires absolus de danger immédiat (remplace danger_front/left relatifs)
   - `length_norm` / `urgency` : contexte temporel
2. **Reward Potential-Based** : bonus de proximité à chaque step (+0.1 × Δmanhattan / CELL)
3. **Budget Étendu** : 5M → 8M timesteps

---

_Informations extraites du code : main.py, PPO.py_
_Date de génération : 2026-03-26_
