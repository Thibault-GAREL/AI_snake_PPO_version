"""
xai_qvalues_ppo.py – Analyse XAI : Politique et Valeur du PPO Snake
====================================================================
Adaptations vs version DQN :
  - Pas de Q-values → probabilités de politique softmax(logits) [4]
  - Valeur d'état supplémentaire issue du critic [1]
  - Visualisations "heatmap" = proba de politique par case de la grille

3 visualisations :
  1. Heatmaps de politique : P(action) par case + heatmap de la valeur d'état
  2. Carte de confiance : gap top1-top2 + politique (flèches)
  3. Évolution temporelle : probabilités des 4 actions + valeur critic

Usage :
    python xai_qvalues_ppo.py --heatmap              # 5 heatmaps (4 actions + valeur)
    python xai_qvalues_ppo.py --gap                  # confiance + politique
    python xai_qvalues_ppo.py --temporal --episodes 2
    python xai_qvalues_ppo.py                        # tout (3 épisodes)
    python xai_qvalues_ppo.py --food-col 8 --food-row 4
    python xai_qvalues_ppo.py --model model_last.pth
"""

import argparse
import math
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
import torch
import torch.nn as nn

from PPO import (
    PPOAgent, SnakeEnv,
    Manager_snake, Snake,
    food as FoodData,
    _dist_north, _dist_south, _dist_west, _dist_east,
    _dist_diag, _food_distances,
)

OUT_DIR = "xai_qvalues_ppo"
os.makedirs(OUT_DIR, exist_ok=True)

def out(filename: str) -> str:
    return os.path.join(OUT_DIR, filename)


# ─── Constantes ────────────────────────────────────────────────────────────────
ACTION_NAMES  = ["UP ↑", "RIGHT →", "DOWN ↓", "LEFT ←"]
ACTION_COLORS = ["#4FC3F7", "#81C784", "#FFB74D", "#F06292"]

GRID_COLS = SnakeEnv.COLS   # 16
GRID_ROWS = SnakeEnv.ROWS   # 8

CMAP_PROB = LinearSegmentedColormap.from_list(
    "prob", ["#0D1B2A", "#1B4F72", "#2E86C1", "#F39C12", "#E74C3C", "#FFFFFF"])
CMAP_VAL  = LinearSegmentedColormap.from_list(
    "val",  ["#C0392B", "#922B21", "#1A1A2E", "#1A5276", "#2E86C1", "#AED6F1"])
CMAP_GAP  = LinearSegmentedColormap.from_list(
    "gap",  ["#1A1A2E", "#16213E", "#0F3460", "#533483", "#E94560"])


# ─── Chargement du modèle ──────────────────────────────────────────────────────
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


# ─── Construction d'un état PPO à partir d'une position de grille ─────────────
def build_state_at_ppo(col: int, row: int,
                        food_col: int, food_row: int,
                        direction: str = "RIGHT") -> np.ndarray:
    """
    Construit un état PPO 22-dim pour (col, row) avec nourriture en (food_col, food_row).
    Serpent de longueur 1, urgence = 0 (début d'épisode).
    """
    W    = SnakeEnv.WIDTH
    H    = SnakeEnv.HEIGHT
    CELL = SnakeEnv.CELL
    M    = SnakeEnv._MAX_DIST

    snake = Manager_snake(W, H)
    snake.add_snake(Snake(col * CELL, row * CELL))
    snake.direction = direction
    f = FoodData(food_col * CELL, food_row * CELL)

    dn  = _dist_north(snake, H)
    ds  = _dist_south(snake, H)
    dw  = _dist_west(snake, W)
    de  = _dist_east(snake, W)
    dne = _dist_diag(snake, W, H, +1, -1)
    dse = _dist_diag(snake, W, H, +1, +1)
    dsw = _dist_diag(snake, W, H, -1, +1)
    dnw = _dist_diag(snake, W, H, -1, -1)

    fn, fne, fe, fse, fsm, fsw, fw, fnw = _food_distances(snake, f)

    DIR_IDX = {"UP": 0, "RIGHT": 1, "DOWN": 2, "LEFT": 3}
    dir_oh  = [0.0, 0.0, 0.0, 0.0]
    dir_oh[DIR_IDX[direction]] = 1.0

    raw = np.array([
        dn, dne, de, dse, ds, dsw, dw, dnw,
        fn, fne, fe, fse, fsm, fsw, fw, fnw,
        *dir_oh,
        0.0,   # longueur normalisée = 0 (serpent de 1)
        0.0,   # urgence = 0
    ], dtype=np.float32)

    raw[:16] /= M
    return raw


def get_policy_and_value(agent: PPOAgent, state: np.ndarray):
    """Retourne (probs [4], value float) pour un état donné."""
    st = torch.tensor(state, dtype=torch.float32,
                      device=agent.device).unsqueeze(0)
    with torch.no_grad():
        logits, value = agent.net(st)
    probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()
    return probs, value.item()


# ─── Scan de la grille ─────────────────────────────────────────────────────────
def scan_grid(agent: PPOAgent, food_col: int, food_row: int):
    """
    Parcourt toutes les cases de la grille (GRID_ROWS × GRID_COLS).
    Retourne :
        policy_probs : [ROWS, COLS, 4]  – proba de politique par action
        best         : [ROWS, COLS]     – action choisie (argmax)
        gap          : [ROWS, COLS]     – top1_prob – top2_prob (confiance)
        value_map    : [ROWS, COLS]     – valeur d'état V(s)
    """
    policy_probs = np.zeros((GRID_ROWS, GRID_COLS, SnakeEnv.ACT_DIM), dtype=np.float32)
    value_map    = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            state = build_state_at_ppo(col, row, food_col, food_row)
            probs, val = get_policy_and_value(agent, state)
            policy_probs[row, col] = probs
            value_map[row, col]    = val

    sorted_p = np.sort(policy_probs, axis=2)
    best     = np.argmax(policy_probs, axis=2)
    gap      = sorted_p[:, :, -1] - sorted_p[:, :, -2]

    return policy_probs, best, gap, value_map


# ─── Visualisation 1 : Heatmaps de politique + valeur ─────────────────────────
def plot_policy_heatmaps(agent: PPOAgent,
                          food_col: int = 8, food_row: int = 4):
    """
    5 heatmaps :
      - 4 heatmaps de probabilité P(action_i) par case
      - 1 heatmap de la valeur d'état V(s)
    La position de la nourriture est marquée d'une étoile.
    Direction par défaut : RIGHT (one-hot [0,1,0,0]).
    """
    print(f"  [Heatmap] Scan de la grille {GRID_ROWS}×{GRID_COLS}…")
    policy_probs, best, gap, value_map = scan_grid(agent, food_col, food_row)

    fig = plt.figure(figsize=(26, 7), facecolor="#0D1117")
    fig.suptitle(
        "Probabilités de politique par case – PPO Snake\n"
        "Nourriture ★ fixe | Direction initiale : RIGHT | Serpent longueur 1",
        fontsize=16, fontweight="bold", color="white", y=1.02
    )
    gs = gridspec.GridSpec(1, 5, figure=fig, wspace=0.32)

    # 4 heatmaps de probabilité (une par action)
    for i, aname in enumerate(ACTION_NAMES):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor("#0D1117")

        im = ax.imshow(policy_probs[:, :, i],
                       cmap=CMAP_PROB, vmin=0.0, vmax=1.0,
                       interpolation="nearest", aspect="auto")
        ax.scatter(food_col, food_row, marker="*", s=350,
                   color="#FFD700", zorder=5, label="Nourriture")

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color="white", labelsize=7)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
        cbar.set_label("P(action)", color="white", fontsize=7)

        ax.set_title(f"{aname}", color=ACTION_COLORS[i],
                     fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Colonne", color="#AAAAAA", fontsize=8)
        ax.set_ylabel("Ligne",   color="#AAAAAA", fontsize=8)
        ax.tick_params(colors="#888888", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333333")

    # Heatmap de la valeur d'état
    ax_v = fig.add_subplot(gs[0, 4])
    ax_v.set_facecolor("#0D1117")
    vabs = max(np.abs(value_map).max(), 0.1)
    im_v = ax_v.imshow(value_map, cmap=CMAP_VAL,
                        vmin=-vabs, vmax=vabs,
                        interpolation="nearest", aspect="auto")
    ax_v.scatter(food_col, food_row, marker="*", s=350,
                 color="#FFD700", zorder=5)
    cbar_v = plt.colorbar(im_v, ax=ax_v, fraction=0.046, pad=0.04)
    cbar_v.ax.yaxis.set_tick_params(color="white", labelsize=7)
    plt.setp(cbar_v.ax.yaxis.get_ticklabels(), color="white")
    cbar_v.set_label("V(s) – valeur", color="white", fontsize=7)
    ax_v.set_title("Valeur d'état V(s)", color="#CE93D8",
                   fontsize=13, fontweight="bold", pad=10)
    ax_v.set_xlabel("Colonne", color="#AAAAAA", fontsize=8)
    ax_v.set_ylabel("Ligne",   color="#AAAAAA", fontsize=8)
    ax_v.tick_params(colors="#888888", labelsize=7)
    for sp in ax_v.spines.values():
        sp.set_edgecolor("#333333")

    plt.tight_layout()
    plt.savefig(out("xai_heatmaps.png"), dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[XAI] Sauvegarde → {out('xai_heatmaps.png')}")
    plt.show()


# ─── Visualisation 2 : Carte de confiance + politique ─────────────────────────
def plot_confidence_map(agent: PPOAgent,
                         food_col: int = 8, food_row: int = 4):
    """
    Gauche : heatmap du gap P_top1 – P_top2 (zones sombres = agent hésitant).
    Droite : politique colorée + flèches d'action.
    """
    print(f"  [Confiance] Scan de la grille {GRID_ROWS}×{GRID_COLS}…")
    policy_probs, best, gap, value_map = scan_grid(agent, food_col, food_row)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="#0D1117")
    fig.suptitle(
        "Confiance de la politique PPO & Carte de décision\n"
        "Gap = P_max – P_2nd  |  Zones sombres = agent hésitant",
        fontsize=15, fontweight="bold", color="white"
    )

    # Gauche : heatmap de confiance
    ax1 = axes[0]
    ax1.set_facecolor("#0D1117")
    im = ax1.imshow(gap, cmap=CMAP_GAP, interpolation="nearest", aspect="auto")
    ax1.scatter(food_col, food_row, marker="*", s=400, color="#FFD700", zorder=5)
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("P_max – P_2nd", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white", labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    ax1.set_title("Confiance (gap de probabilités)", color="white",
                  fontsize=13, pad=10)
    ax1.set_xlabel("Colonne", color="#AAAAAA", fontsize=9)
    ax1.set_ylabel("Ligne",   color="#AAAAAA", fontsize=9)
    ax1.tick_params(colors="#888888", labelsize=7)
    for sp in ax1.spines.values():
        sp.set_edgecolor("#333333")

    # Droite : politique colorée + flèches
    ax2 = axes[1]
    ax2.set_facecolor("#0D1117")

    color_table = {
        0: np.array([0.31, 0.76, 0.97]),   # UP    bleu
        1: np.array([0.51, 0.78, 0.52]),   # RIGHT vert
        2: np.array([1.00, 0.72, 0.30]),   # DOWN  orange
        3: np.array([0.94, 0.38, 0.57]),   # LEFT  rose
    }
    policy_rgb = np.zeros((GRID_ROWS, GRID_COLS, 3))
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            policy_rgb[r, c] = color_table[best[r, c]]
    gap_norm = (gap - gap.min()) / (gap.max() - gap.min() + 1e-8)
    alpha    = 0.30 + 0.70 * gap_norm
    for ch in range(3):
        policy_rgb[:, :, ch] *= alpha

    ax2.imshow(policy_rgb, interpolation="nearest", aspect="auto")

    arrows = {0: (0, -0.35), 1: (0.35, 0), 2: (0, 0.35), 3: (-0.35, 0)}
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            dx, dy = arrows[best[r, c]]
            ax2.annotate(
                "", xy=(c + dx, r + dy), xytext=(c, r),
                arrowprops=dict(arrowstyle="->", color="white", lw=0.8),
            )
    ax2.scatter(food_col, food_row, marker="*", s=400, color="#FFD700", zorder=5)

    legend_p = [mpatches.Patch(color=tuple(color_table[i]), label=ACTION_NAMES[i])
                for i in range(4)]
    ax2.legend(handles=legend_p, loc="upper right", fontsize=8,
               facecolor="#1A1A2E", edgecolor="#444", labelcolor="white")
    ax2.set_title("Politique apprise (action par case)", color="white",
                  fontsize=13, pad=10)
    ax2.set_xlabel("Colonne", color="#AAAAAA", fontsize=9)
    ax2.set_ylabel("Ligne",   color="#AAAAAA", fontsize=9)
    ax2.tick_params(colors="#888888", labelsize=7)
    for sp in ax2.spines.values():
        sp.set_edgecolor("#333333")

    plt.tight_layout()
    plt.savefig(out("xai_confidence.png"), dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[XAI] Sauvegarde → {out('xai_confidence.png')}")
    plt.show()


# ─── Visualisation 3 : Évolution temporelle ───────────────────────────────────
def plot_temporal(agent: PPOAgent, num_episodes: int = 3):
    """
    Lance N épisodes en mode greedy et enregistre à chaque step :
      - Les 4 probabilités de politique (softmax des logits)
      - La valeur d'état V(s) du critic
    Affiche aussi : nourriture mangée (|) et mort (×).
    """
    env = SnakeEnv(render=False)
    all_episodes = []

    for ep in range(num_episodes):
        obs   = env.reset()
        done  = False
        ep_data = {"probs": [], "values": [], "events": [], "scores": []}
        step       = 0
        prev_score = 0

        while not done:
            probs, val = get_policy_and_value(agent, obs)
            action     = int(np.argmax(probs))

            ep_data["probs"].append(probs.copy())
            ep_data["values"].append(val)

            obs, reward, done, info = env.step(action)

            if info["score"] > prev_score:
                ep_data["events"].append((step, "food"))
                prev_score = info["score"]
            if done and reward < -5.0:   # pénalité de mort < -10
                ep_data["events"].append((step, "death"))

            ep_data["scores"].append(info["score"])
            step += 1

        all_episodes.append(ep_data)
        print(f"[XAI] Épisode {ep+1}/{num_episodes} – "
              f"score {info['score']}  ({step} steps)")

    env.close()

    fig, axes = plt.subplots(
        num_episodes, 1,
        figsize=(18, 6 * num_episodes),
        facecolor="#0D1117",
        squeeze=False,
    )
    fig.suptitle(
        "Évolution temporelle – Probabilités de politique + Valeur d'état (PPO)\n"
        "Courbes solides = P(action)  |  Tirets blancs = V(s) critic  |  "
        "| = nourriture  ✕ = mort",
        fontsize=14, fontweight="bold", color="white", y=1.01
    )

    for ep_idx, ep_data in enumerate(all_episodes):
        ax = axes[ep_idx, 0]
        ax.set_facecolor("#0D1B2A")

        probs_arr  = np.array(ep_data["probs"])    # [T, 4]
        values_arr = np.array(ep_data["values"])   # [T]
        T          = len(probs_arr)
        steps      = np.arange(T)

        # Probabilités des 4 actions
        for i in range(SnakeEnv.ACT_DIM):
            ax.fill_between(steps, probs_arr[:, i], alpha=0.08,
                            color=ACTION_COLORS[i])
            ax.plot(steps, probs_arr[:, i],
                    label=ACTION_NAMES[i], color=ACTION_COLORS[i],
                    lw=1.4, alpha=0.9)

        # Valeur d'état (axe secondaire ou normalisée [0,1] pour lisibilité)
        v_min, v_max = values_arr.min(), values_arr.max()
        v_range = v_max - v_min + 1e-8
        v_norm  = (values_arr - v_min) / v_range   # → [0, 1]
        ax.plot(steps, v_norm, color="white", lw=1.0,
                ls="--", alpha=0.55, label=f"V(s) norm. [{v_min:.1f}…{v_max:.1f}]")

        # Événements
        ylim_top = 1.05
        for step_ev, ev_type in ep_data["events"]:
            if ev_type == "food":
                ax.axvline(x=step_ev, color="#2ECC71", lw=1.5, ls=":", alpha=0.8)
                ax.text(step_ev + 0.5, ylim_top * 0.94, "●",
                        fontsize=9, color="#2ECC71", alpha=0.9)
            elif ev_type == "death":
                ax.axvline(x=step_ev, color="#E74C3C", lw=2.0, ls="-", alpha=0.9)
                ax.text(step_ev + 0.5, ylim_top * 0.94, "✕",
                        fontsize=9, color="#E74C3C")

        score_final = ep_data["scores"][-1] if ep_data["scores"] else 0
        ax.set_title(
            f"Épisode {ep_idx+1}  –  Score final : {score_final}  |  Steps : {T}",
            color="white", fontsize=12, pad=8
        )
        ax.set_xlabel("Step", color="#AAAAAA", fontsize=9)
        ax.set_ylabel("Probabilité / Valeur norm.", color="#AAAAAA", fontsize=9)
        ax.set_ylim(-0.05, ylim_top)
        ax.tick_params(colors="#888888", labelsize=8)
        ax.legend(loc="upper left", fontsize=8, facecolor="#0D1117",
                  edgecolor="#444444", labelcolor="white",
                  framealpha=0.8, ncol=5)
        ax.grid(axis="y", color="#1E3A5F", lw=0.5, alpha=0.6)
        for sp in ax.spines.values():
            sp.set_edgecolor("#1E3A5F")

    plt.tight_layout()
    plt.savefig(out("xai_temporal.png"), dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[XAI] Sauvegarde → {out('xai_temporal.png')}")
    plt.show()


# ─── Point d'entrée ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="XAI – Politique et Valeur PPO Snake")
    parser.add_argument("--heatmap",  action="store_true",
                        help="Heatmaps P(action) + V(s) sur la grille")
    parser.add_argument("--gap",      action="store_true",
                        help="Carte de confiance + politique avec flèches")
    parser.add_argument("--temporal", action="store_true",
                        help="Évolution temporelle probas + valeur critic")
    parser.add_argument("--model",    type=str, default="model_best.pth")
    parser.add_argument("--food-col", type=int, default=8,
                        help="Colonne de la nourriture pour heatmap/gap (défaut : 8)")
    parser.add_argument("--food-row", type=int, default=4,
                        help="Ligne de la nourriture pour heatmap/gap (défaut : 4)")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Épisodes pour analyse temporelle (défaut : 3)")
    args = parser.parse_args()

    run_all = not (args.heatmap or args.gap or args.temporal)
    agent   = load_agent(args.model)

    if run_all or args.heatmap:
        print("\n[XAI] ── Heatmaps de politique ──")
        plot_policy_heatmaps(agent, food_col=args.food_col, food_row=args.food_row)

    if run_all or args.gap:
        print("\n[XAI] ── Carte de confiance ──")
        plot_confidence_map(agent, food_col=args.food_col, food_row=args.food_row)

    if run_all or args.temporal:
        print(f"\n[XAI] ── Évolution temporelle ({args.episodes} épisodes) ──")
        plot_temporal(agent, num_episodes=args.episodes)

    print("\n[XAI] Analyse terminée.")


if __name__ == "__main__":
    main()

# ── Tests recommandés ──────────────────────────────────────────────────────────
# Test heatmap (~10s) :
#   python xai_qvalues_ppo.py --heatmap
#   → Vérifie : xai_qvalues_ppo/xai_heatmaps.png
#     ✓ 5 subplots côte à côte (4 actions + valeur d'état)
#     ✓ Étoile jaune = position de la nourriture
#     ✓ Zones chaudes = l'agent préfère cette action depuis ces cases
#     ✓ Heatmap valeur : zones bleues près de la nourriture (V(s) élevé)
#
# Test confiance (~10s) :
#   python xai_qvalues_ppo.py --gap
#   → Vérifie : xai_qvalues_ppo/xai_confidence.png
#     ✓ Zones sombres = agent hésitant (gap faible entre 2 actions)
#     ✓ Flèches cohérentes avec la direction vers la nourriture
#
# Test temporel (~30s) :
#   python xai_qvalues_ppo.py --temporal --episodes 2
#   → Vérifie : xai_qvalues_ppo/xai_temporal.png
#     ✓ 4 courbes de probabilité (somme ≈ 1 à chaque step)
#     ✓ Courbe blanche tiretée = valeur critic normalisée
#     ✓ Pics verts = nourriture mangée, croix rouges = mort
#
# Changer la position de la nourriture :
#   python xai_qvalues_ppo.py --heatmap --food-col 2 --food-row 2
