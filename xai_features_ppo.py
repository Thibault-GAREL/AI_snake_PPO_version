"""
xai_features_ppo.py – Analyse XAI : Feature Importance du PPO Snake
====================================================================
3 analyses :
  1. Permutation Importance  : brouiller chaque feature → mesurer la chute de score
  2. Variance des poids W1   : couche shared[0] Linear(22→256)
  3. Corrélation features/actions : quelle feature déclenche quelle action ?

Les 22 features PPO :
  [0:8]   Distances aux dangers (N NE E SE S SW W NW), normalisées
  [8:16]  Distances à la nourriture (mêmes 8 directions), normalisées
  [16:20] One-hot direction courante (UP RIGHT DOWN LEFT)
  [20]    Longueur du serpent normalisée (0→1)
  [21]    Urgence nourriture : steps_since_food / MAX_STEPS (0→1)

Usage :
    python xai_features_ppo.py --variance              # instantané, sans épisodes
    python xai_features_ppo.py --permutation --episodes 20
    python xai_features_ppo.py --correlation --episodes 20
    python xai_features_ppo.py                         # tout (20 épisodes)
"""

import argparse
import math
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches
from scipy.stats import pearsonr
import torch
import torch.nn as nn

from PPO import PPOAgent, SnakeEnv

OUT_DIR = "xai_features_ppo"
os.makedirs(OUT_DIR, exist_ok=True)

def out(filename: str) -> str:
    return os.path.join(OUT_DIR, filename)


# ─── Noms des 22 features ──────────────────────────────────────────────────────
FEATURE_NAMES = [
    # Danger distances (continues) [0:8]
    "Danger dist N",  "Danger dist NE", "Danger dist E",  "Danger dist SE",
    "Danger dist S",  "Danger dist SW", "Danger dist W",  "Danger dist NW",
    # Food distances (continues) [8:16]
    "Food dist N",    "Food dist NE",   "Food dist E",    "Food dist SE",
    "Food dist S",    "Food dist SW",   "Food dist W",    "Food dist NW",
    # Food delta (continu) [16:18]
    "Food delta X",   "Food delta Y",
    # Danger binaire immédiat N/E/S/W [18:22]
    "Danger bin N",   "Danger bin E",   "Danger bin S",   "Danger bin W",
    # Direction one-hot [22:26]
    "Dir UP",         "Dir RIGHT",      "Dir DOWN",       "Dir LEFT",
    # Contexte [26:28]
    "Longueur",       "Urgence food",
]
N_FEATURES = len(FEATURE_NAMES)   # 28

# Catégories pour la colorisation
# 0-7 : danger dist, 8-15 : food dist, 16-17 : food delta,
# 18-21 : danger binaire, 22-25 : direction, 26-27 : contexte
def _feat_color(i: int) -> str:
    if i < 8:   return "#4FC3F7"   # bleu     → danger distances
    if i < 16:  return "#FFB74D"   # orange   → food distances
    if i < 18:  return "#FFF176"   # jaune    → food delta
    if i < 22:  return "#EF5350"   # rouge    → danger binaire
    if i < 26:  return "#81C784"   # vert     → direction
    return              "#F06292"  # rose     → contexte

FEAT_COLORS = [_feat_color(i) for i in range(N_FEATURES)]

ACTION_NAMES  = ["UP ↑", "RIGHT →", "DOWN ↓", "LEFT ←"]
ACTION_COLORS = ["#4FC3F7", "#81C784", "#FFB74D", "#F06292"]

BG       = "#0D1117"
PANEL_BG = "#0D1B2A"
GRID_COL = "#1E3A5F"
TEXT_COL = "#CCDDEE"

CMAP_IMPORTANCE = LinearSegmentedColormap.from_list(
    "imp", ["#0D1B2A", "#1A3A5C", "#1F618D", "#2E86C1", "#F39C12", "#E74C3C"])
CMAP_CORR = LinearSegmentedColormap.from_list(
    "corr", ["#C0392B", "#922B21", "#1A1A2E", "#1A5276", "#2E86C1"])
CMAP_VAR = LinearSegmentedColormap.from_list(
    "var", ["#0D1B2A", "#154360", "#1F618D", "#AED6F1", "#EBF5FB"])


# ─── Utilitaires ───────────────────────────────────────────────────────────────
def load_agent(path: str = "model_best.pth") -> PPOAgent:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent  = PPOAgent(obs_dim=SnakeEnv.OBS_DIM, act_dim=SnakeEnv.ACT_DIM, device=device)
    try:
        agent.load(path)
        print(f"[XAI] Modèle chargé ← {path}")
    except FileNotFoundError:
        print(f"[WARN] {path} introuvable → poids aléatoires.")
    agent.net.eval()
    return agent


def run_episode(agent: PPOAgent, env: SnakeEnv,
                shuffle_feature: int = -1) -> tuple:
    """
    Joue un épisode en mode greedy (argmax logits).
    shuffle_feature : si ≥ 0, remplace cette feature par une valeur uniforme aléatoire.
    Retourne (score, states_log, actions_log).
    """
    obs  = env.reset()
    done = False
    states_log  = []
    actions_log = []

    while not done:
        s = obs.copy()
        if shuffle_feature >= 0:
            s[shuffle_feature] = float(np.random.uniform(0.0, 1.0))

        obs_t = torch.tensor(s, dtype=torch.float32,
                             device=agent.device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = agent.net(obs_t)
        action = int(logits.argmax(dim=1).item())

        states_log.append(s)
        actions_log.append(action)
        obs, _, done, info = env.step(action)

    return info["score"], states_log, actions_log


def apply_style(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    ax.set_facecolor(PANEL_BG)
    if title:  ax.set_title(title,  color="white",  fontsize=12, fontweight="bold", pad=10)
    if xlabel: ax.set_xlabel(xlabel, color=TEXT_COL, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=TEXT_COL, fontsize=9)
    ax.tick_params(colors="#8899AA", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_COL)
    ax.grid(axis="x", color=GRID_COL, lw=0.5, alpha=0.5)


# ─── Analyse 1 : Permutation Importance ───────────────────────────────────────
def compute_permutation_importance(agent, env, n_episodes: int = 20):
    # Baseline
    print(f"  [PI] Calcul du baseline ({n_episodes} épisodes)…")
    baseline_scores = [run_episode(agent, env)[0] for _ in range(n_episodes)]
    baseline_mean   = float(np.mean(baseline_scores))
    print(f"  [PI] Score baseline moyen : {baseline_mean:.2f}")

    drops     = np.zeros(N_FEATURES)
    drops_std = np.zeros(N_FEATURES)

    for fi in range(N_FEATURES):
        shuffled  = [run_episode(agent, env, shuffle_feature=fi)[0]
                     for _ in range(n_episodes)]
        mean_sh   = float(np.mean(shuffled))
        drop      = baseline_mean - mean_sh
        drops[fi]     = max(drop, 0.0)
        drops_std[fi] = float(np.std(shuffled))
        print(f"  [PI] [{fi:>2}] {FEATURE_NAMES[fi]:<18} : "
              f"score={mean_sh:.2f}  drop={drop:+.2f}")

    return drops, baseline_mean, drops_std


def plot_permutation_importance(drops, baseline, drops_std):
    n     = N_FEATURES
    order = np.argsort(drops)[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(20, 10), facecolor=BG,
                             gridspec_kw={"width_ratios": [2, 1]})
    fig.suptitle(
        f"Permutation Importance – PPO Snake  (baseline score : {baseline:.2f})",
        fontsize=16, fontweight="bold", color="white"
    )

    # Gauche : barplot horizontal
    ax = axes[0]
    ax.set_facecolor(PANEL_BG)
    norm   = Normalize(vmin=drops.min(), vmax=drops.max())
    colors = [CMAP_IMPORTANCE(norm(drops[order[i]])) for i in range(n)]

    ax.barh(range(n), drops[order],
            xerr=drops_std[order], color=colors, edgecolor="#1A1A2E",
            error_kw=dict(ecolor="#AAAAAA", lw=1.2, capsize=3),
            height=0.72)

    for i, (drop, std) in enumerate(zip(drops[order], drops_std[order])):
        ax.text(drop + std + 0.01, i, f"{drop:.2f}",
                va="center", ha="left", color=TEXT_COL, fontsize=8)

    ax.set_yticks(range(n))
    ax.set_yticklabels([FEATURE_NAMES[i] for i in order], color=TEXT_COL, fontsize=8.5)

    # Séparateurs catégories
    def _sep(threshold, label, y_off=0):
        count = sum(1 for i in order if i < threshold)
        ax.axhline(y=n - count - 0.5, color="#F39C12", lw=1.0,
                   ls="--", alpha=0.6)
    _sep(8,  "danger/food")
    _sep(16, "food/dir")
    _sep(20, "dir/ctx")

    apply_style(ax, title="Chute de score par feature brouillée",
                xlabel="Drop de score moyen (baseline – brouillée)")

    # Légende catégories
    patches = [
        mpatches.Patch(color="#4FC3F7", label="Danger (0–7)"),
        mpatches.Patch(color="#FFB74D", label="Nourriture (8–15)"),
        mpatches.Patch(color="#81C784", label="Direction (16–19)"),
        mpatches.Patch(color="#F06292", label="Contexte (20–21)"),
    ]
    ax.legend(handles=patches, fontsize=8, facecolor="#0D1117",
              edgecolor="#444", labelcolor="white", loc="lower right")

    # Droite : radar des top-8
    ax2 = axes[1]
    ax2.set_facecolor(PANEL_BG)
    ax2.set_aspect("equal")
    top8        = order[:8]
    drops_top8  = drops[top8]
    vals        = drops_top8 / (drops_top8.max() + 1e-8)
    labs        = [FEATURE_NAMES[i] for i in top8]
    N_rad       = len(top8)
    angles      = [2 * math.pi * k / N_rad for k in range(N_rad)] + [0]
    vals_r      = list(vals) + [vals[0]]

    for level in [0.25, 0.5, 0.75, 1.0]:
        ring_x = [level * math.cos(a) for a in angles]
        ring_y = [level * math.sin(a) for a in angles]
        ax2.plot(ring_x, ring_y, color=GRID_COL, lw=0.7, alpha=0.6)
        ax2.text(level + 0.04, 0.02, f"{int(level*100)}%",
                 color="#7A9CC0", fontsize=6, va="center", alpha=0.8)
    for a in angles[:-1]:
        ax2.plot([0, math.cos(a)], [0, math.sin(a)],
                 color=GRID_COL, lw=0.7, alpha=0.6)

    xs = [v * math.cos(a) for v, a in zip(vals_r, angles)]
    ys = [v * math.sin(a) for v, a in zip(vals_r, angles)]
    ax2.fill(xs, ys, color="#2E86C1", alpha=0.28)
    ax2.plot(xs, ys, color="#4FC3F7", lw=2.2)
    ax2.scatter(xs[:-1], ys[:-1], color="#FFD700", s=70, zorder=5)

    for rank, (fi, a, lab) in enumerate(zip(top8, angles[:-1], labs)):
        drop_val = drops_top8[rank]
        lab_col  = _feat_color(fi)
        ax2.text(1.38 * math.cos(a), 1.38 * math.sin(a),
                 f"#{rank+1} {lab}\ndrop={drop_val:.2f}",
                 ha="center", va="center", color=lab_col, fontsize=7,
                 fontweight="bold", multialignment="center")

    ax2.set_xlim(-1.75, 1.75)
    ax2.set_ylim(-2.20, 1.75)
    ax2.axis("off")
    ax2.set_title("Top 8 features – Radar\n(rayon ↔ chute de score si brouillée)",
                  color="white", fontsize=11, fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(out("xai_permutation.png"), dpi=150,
                bbox_inches="tight", facecolor=BG)
    print(f"[XAI] Sauvegarde → {out('xai_permutation.png')}")
    plt.show()


# ─── Analyse 2 : Variance des poids (couche shared[0]) ────────────────────────
def compute_weight_variance(agent: PPOAgent):
    """
    Analyse la couche Linear(22→256) = agent.net.shared[0].
    Retourne (l2_norms [22], stds [22], W [256, 22]).
    """
    W      = agent.net.shared[0].weight.detach().cpu().numpy()   # [256, 22]
    l2_norms = np.linalg.norm(W, axis=0)   # norme L2 par feature d'entrée
    stds     = W.std(axis=0)
    return l2_norms, stds, W


def plot_weight_variance(l2_norms, stds, W):
    fig = plt.figure(figsize=(22, 11), facecolor=BG)
    fig.suptitle(
        "Analyse des poids – couche shared[0] Linear(22→256) – PPO\n"
        "Features avec forte norme L2 / fort écart-type = très utilisées par le réseau",
        fontsize=16, fontweight="bold", color="white"
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.35, hspace=0.45)

    # Haut gauche : L2 norm
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(PANEL_BG)
    order1 = np.argsort(l2_norms)[::-1]
    norm_c = Normalize(vmin=l2_norms.min(), vmax=l2_norms.max())
    colors1 = [CMAP_VAR(norm_c(l2_norms[i])) for i in order1]
    ax1.barh(range(N_FEATURES), l2_norms[order1], color=colors1,
             edgecolor="#0D1117", height=0.7)
    ax1.set_yticks(range(N_FEATURES))
    ax1.set_yticklabels([FEATURE_NAMES[i] for i in order1], color=TEXT_COL, fontsize=8)
    for k, v in enumerate(l2_norms[order1]):
        color_y = _feat_color(order1[k])
        ax1.get_yticklabels()[k].set_color(color_y)
        ax1.text(v + 0.002, k, f"{v:.3f}", va="center", color=TEXT_COL, fontsize=7)
    apply_style(ax1, title="Norme L2 par feature d'entrée",
                xlabel="‖W[:,i]‖₂ – importance structurelle")

    # Haut droite : std
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(PANEL_BG)
    order2  = np.argsort(stds)[::-1]
    colors2 = [CMAP_IMPORTANCE(norm_c(stds[i])) for i in order2]
    ax2.barh(range(N_FEATURES), stds[order2], color=colors2,
             edgecolor="#0D1117", height=0.7)
    ax2.set_yticks(range(N_FEATURES))
    ax2.set_yticklabels([FEATURE_NAMES[i] for i in order2], color=TEXT_COL, fontsize=8)
    for k, v in enumerate(stds[order2]):
        ax2.get_yticklabels()[k].set_color(_feat_color(order2[k]))
        ax2.text(v + 0.0005, k, f"{v:.4f}", va="center", color=TEXT_COL, fontsize=7)
    apply_style(ax2, title="Écart-type des poids par feature",
                xlabel="std(W[:,i]) – dispersion des connexions")

    # Bas gauche : heatmap W (64 premiers neurones × 22 features)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(PANEL_BG)
    W_show = W[:64, :]
    vabs   = np.abs(W_show).max()
    im = ax3.imshow(W_show.T, cmap="RdBu_r", vmin=-vabs, vmax=vabs,
                    aspect="auto", interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label("Valeur du poids", color=TEXT_COL, fontsize=8)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COL, labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COL)
    ax3.set_yticks(range(N_FEATURES))
    ax3.set_yticklabels(FEATURE_NAMES, color=TEXT_COL, fontsize=7)
    for k, tick in enumerate(ax3.get_yticklabels()):
        tick.set_color(_feat_color(k))
    ax3.set_xlabel("Neurone caché (64 premiers / 256)", color=TEXT_COL, fontsize=8)
    ax3.set_title("Matrice poids W₁ (features × neurones cachés)",
                  color="white", fontsize=11, fontweight="bold", pad=8)
    ax3.tick_params(colors="#8899AA", labelsize=7)
    for sp in ax3.spines.values():
        sp.set_edgecolor(GRID_COL)

    # Séparateurs catégories
    for sep in [7.5, 15.5, 19.5]:
        ax3.axhline(y=sep, color="#F39C12", lw=1.0, ls="--", alpha=0.6)

    # Bas droite : scatter L2 vs std
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(PANEL_BG)
    sc = ax4.scatter(l2_norms, stds, c=FEAT_COLORS, s=90,
                     edgecolors="#222244", lw=0.8, zorder=3)
    for i, (x, y) in enumerate(zip(l2_norms, stds)):
        ax4.annotate(FEATURE_NAMES[i], (x, y),
                     textcoords="offset points", xytext=(5, 3),
                     color=TEXT_COL, fontsize=6.5, alpha=0.85)
    ax4.axvline(x=np.median(l2_norms), color="#F39C12", ls="--", lw=1, alpha=0.6)
    ax4.axhline(y=np.median(stds),     color="#F39C12", ls="--", lw=1, alpha=0.6)
    ax4.text(np.median(l2_norms) * 0.3, stds.max() * 0.93,
             "faible\nimportance", color="#F39C12", fontsize=7.5, alpha=0.7)
    ax4.text(np.median(l2_norms) * 1.06, stds.max() * 0.93,
             "forte\nimportance", color="#F39C12", fontsize=7.5, alpha=0.7)
    patches = [
        mpatches.Patch(color="#4FC3F7", label="Danger (0–7)"),
        mpatches.Patch(color="#FFB74D", label="Nourriture (8–15)"),
        mpatches.Patch(color="#81C784", label="Direction (16–19)"),
        mpatches.Patch(color="#F06292", label="Contexte (20–21)"),
    ]
    ax4.legend(handles=patches, fontsize=8, facecolor="#0D1117",
               edgecolor="#444", labelcolor="white")
    apply_style(ax4, title="Nuage L2-norm vs Std – quadrant d'importance",
                xlabel="Norme L2", ylabel="Écart-type")

    plt.savefig(out("xai_variance.png"), dpi=150,
                bbox_inches="tight", facecolor=BG)
    print(f"[XAI] Sauvegarde → {out('xai_variance.png')}")
    plt.show()


# ─── Analyse 3 : Corrélation features / actions ───────────────────────────────
def compute_feature_action_correlation(agent, env, n_episodes: int = 20):
    all_states  = []
    all_actions = []

    print(f"  [Corr] Collecte sur {n_episodes} épisodes…")
    for ep in range(n_episodes):
        score, states, actions = run_episode(agent, env)
        all_states.extend(states)
        all_actions.extend(actions)

    states_arr  = np.array(all_states,  dtype=np.float32)   # [T, 22]
    actions_arr = np.array(all_actions, dtype=np.int32)      # [T]
    print(f"  [Corr] {len(actions_arr)} transitions collectées.")

    corr_matrix = np.zeros((N_FEATURES, SnakeEnv.ACT_DIM))
    for fi in range(N_FEATURES):
        for ai in range(SnakeEnv.ACT_DIM):
            binary = (actions_arr == ai).astype(float)
            r, _   = pearsonr(states_arr[:, fi], binary)
            corr_matrix[fi, ai] = r if not np.isnan(r) else 0.0

    mean_per_action = np.zeros((SnakeEnv.ACT_DIM, N_FEATURES))
    std_per_action  = np.zeros((SnakeEnv.ACT_DIM, N_FEATURES))
    for ai in range(SnakeEnv.ACT_DIM):
        mask = actions_arr == ai
        if mask.sum() > 0:
            mean_per_action[ai] = states_arr[mask].mean(axis=0)
            std_per_action[ai]  = states_arr[mask].std(axis=0)

    return corr_matrix, mean_per_action, std_per_action


def plot_feature_action_correlation(corr_matrix, mean_per_action, std_per_action):
    fig = plt.figure(figsize=(22, 14), facecolor=BG)
    fig.suptitle(
        "Corrélation Features → Actions – PPO Snake\n"
        "Ce qui déclenche chaque décision de l'agent",
        fontsize=16, fontweight="bold", color="white"
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.4, hspace=0.5)

    # Heatmap centrale [22 × 4]
    ax_heat = fig.add_subplot(gs[:, 0])
    ax_heat.set_facecolor(PANEL_BG)
    vabs = np.abs(corr_matrix).max()
    im   = ax_heat.imshow(corr_matrix, cmap=CMAP_CORR,
                           vmin=-vabs, vmax=vabs,
                           aspect="auto", interpolation="nearest")
    for fi in range(N_FEATURES):
        for ai in range(SnakeEnv.ACT_DIM):
            v = corr_matrix[fi, ai]
            c = "white" if abs(v) > 0.12 else "#888888"
            ax_heat.text(ai, fi, f"{v:+.2f}", ha="center", va="center",
                         color=c, fontsize=7.5, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.set_label("Corrélation de Pearson", color=TEXT_COL, fontsize=8)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COL, labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COL)
    ax_heat.set_xticks(range(SnakeEnv.ACT_DIM))
    ax_heat.set_xticklabels(ACTION_NAMES, color=TEXT_COL, fontsize=9)
    ax_heat.set_yticks(range(N_FEATURES))
    ax_heat.set_yticklabels(FEATURE_NAMES, fontsize=7.5)
    for k, tick in enumerate(ax_heat.get_yticklabels()):
        tick.set_color(_feat_color(k))
    # Séparateurs catégories
    for sep in [7.5, 15.5, 19.5]:
        ax_heat.axhline(y=sep, color="#F39C12", lw=1.2, ls="--", alpha=0.7)
    ax_heat.set_title("Corrélation\nfeature × action", color="white",
                      fontsize=12, fontweight="bold", pad=10)
    for sp in ax_heat.spines.values():
        sp.set_edgecolor(GRID_COL)

    # 4 barplots par action
    positions = [(0, 1), (0, 2), (1, 1), (1, 2)]
    for ai, (row, col) in enumerate(positions):
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor(PANEL_BG)
        vals  = corr_matrix[:, ai]
        order = np.argsort(np.abs(vals))[::-1]
        bar_c = [ACTION_COLORS[ai] if v >= 0 else "#E74C3C" for v in vals[order]]
        ax.barh(list(range(N_FEATURES)), vals[order],
                color=bar_c, edgecolor="#0D1117", alpha=0.85, height=0.7)
        ax.axvline(x=0, color="#AAAAAA", lw=1.0, alpha=0.5)
        ax.set_yticks(list(range(N_FEATURES)))
        ax.set_yticklabels([FEATURE_NAMES[i] for i in order], fontsize=7)
        for k, tick in enumerate(ax.get_yticklabels()):
            tick.set_color(_feat_color(order[k]))
        ax.set_xlim(-vabs * 1.2, vabs * 1.2)
        apply_style(ax, xlabel="Corrélation de Pearson")
        ax.set_title(f"Action : {ACTION_NAMES[ai]}",
                     color=ACTION_COLORS[ai], fontsize=11,
                     fontweight="bold", pad=8)

    plt.savefig(out("xai_correlation.png"), dpi=150,
                bbox_inches="tight", facecolor=BG)
    print(f"[XAI] Sauvegarde → {out('xai_correlation.png')}")
    plt.show()

    # Bonus : profil sensoriel par action
    _plot_mean_per_action(mean_per_action, std_per_action)


def _plot_mean_per_action(mean_per_action, std_per_action):
    fig, axes = plt.subplots(1, SnakeEnv.ACT_DIM, figsize=(28, 10), facecolor=BG)
    fig.suptitle(
        "Profil sensoriel par action – PPO Snake\n"
        "Valeur moyenne de chaque feature (22) quand l'agent choisit cette action",
        fontsize=12, fontweight="bold", color="white", y=1.01
    )
    ypos = np.arange(N_FEATURES)

    for ai, ax in enumerate(axes):
        ax.set_facecolor(PANEL_BG)
        means = mean_per_action[ai]
        stds  = std_per_action[ai]

        for k in range(N_FEATURES):
            ax.axhspan(k - 0.5, k + 0.5,
                       color="#0F2233" if k % 2 == 0 else PANEL_BG,
                       alpha=0.5, zorder=0)

        ax.barh(ypos, means, xerr=stds,
                color=ACTION_COLORS[ai], alpha=0.82,
                edgecolor="#0D1117", height=0.65, zorder=2,
                error_kw=dict(ecolor="#AAAAAA", lw=1, capsize=3))

        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(m + s + 0.01, i, f"{m:.2f}",
                    va="center", ha="left", color=TEXT_COL, fontsize=6.5)

        ax.set_yticks(ypos)
        ax.set_yticklabels(FEATURE_NAMES, fontsize=7.5)
        for k, tick in enumerate(ax.get_yticklabels()):
            tick.set_color(_feat_color(k))

        # Séparateurs catégories
        for sep in [7.5, 15.5, 19.5]:
            ax.axhline(y=sep, color="#F39C12", lw=1.2, ls="--", alpha=0.7, zorder=3)

        ax.set_title(f"Action : {ACTION_NAMES[ai]}",
                     color=ACTION_COLORS[ai], fontsize=12,
                     fontweight="bold", pad=10)
        ax.set_xlabel("Valeur normalisée [0→1]", color=TEXT_COL, fontsize=8)
        ax.set_xlim(0, 1.15)
        ax.tick_params(axis="x", colors="#8899AA", labelsize=8)
        ax.tick_params(axis="y", colors="#8899AA", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID_COL)
        ax.grid(axis="x", color=GRID_COL, lw=0.5, alpha=0.5, zorder=1)

    plt.tight_layout()
    plt.savefig(out("xai_mean_per_action.png"), dpi=150,
                bbox_inches="tight", facecolor=BG)
    print(f"[XAI] Sauvegarde → {out('xai_mean_per_action.png')}")
    plt.show()


# ─── Point d'entrée ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="XAI – Feature Importance PPO Snake")
    parser.add_argument("--permutation", action="store_true")
    parser.add_argument("--variance",    action="store_true")
    parser.add_argument("--correlation", action="store_true")
    parser.add_argument("--model",    type=str, default="model_best.pth")
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    run_all = not (args.permutation or args.variance or args.correlation)
    agent   = load_agent(args.model)
    env     = SnakeEnv(render=False)

    if run_all or args.variance:
        print("\n[XAI] ── Variance des poids (shared[0]) ──")
        l2, stds, W = compute_weight_variance(agent)
        plot_weight_variance(l2, stds, W)

    if run_all or args.permutation:
        print(f"\n[XAI] ── Permutation Importance ({args.episodes} épisodes) ──")
        drops, baseline, drops_std = compute_permutation_importance(
            agent, env, args.episodes)
        plot_permutation_importance(drops, baseline, drops_std)

    if run_all or args.correlation:
        print(f"\n[XAI] ── Corrélation features×actions ({args.episodes} épisodes) ──")
        corr, means, stds_a = compute_feature_action_correlation(
            agent, env, args.episodes)
        plot_feature_action_correlation(corr, means, stds_a)

    env.close()
    print("\n[XAI] Analyse terminée.")


if __name__ == "__main__":
    main()

# ── Tests recommandés ──────────────────────────────────────────────────────────
# Test instantané (pas d'épisodes) :
#   python xai_features_ppo.py --variance
#   → Vérifie : xai_features_ppo/xai_variance.png
#     ✓ 22 features sur l'axe Y (colorées par catégorie)
#     ✓ Heatmap des poids W₁ [22 × 64]
#     ✓ Nuage L2 vs Std avec quadrant d'importance
#
# Test corrélation (~1min) :
#   python xai_features_ppo.py --correlation --episodes 20
#   → Vérifie : xai_features_ppo/xai_correlation.png
#                xai_features_ppo/xai_mean_per_action.png
#     ✓ Heatmap corrélation [22 features × 4 actions]
#     ✓ 4 profils sensoriels par action
#
# Test permutation (lent, ~5-10min) :
#   python xai_features_ppo.py --permutation --episodes 20
#   → Vérifie : xai_features_ppo/xai_permutation.png
#     ✓ Features les plus importantes ont les barres les plus longues
#     ✓ Radar top-8
