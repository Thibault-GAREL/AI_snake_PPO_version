"""
xai_shap_ppo.py – Analyse XAI : SHAP pour le PPO Snake
=======================================================
Adaptations vs version DQN :
  - Wrapper ActorWrapper expose uniquement les logits (actor head) pour DeepExplainer
  - 22 features (vs 16 DQN) avec FEATURE_NAMES mis à jour
  - Imports depuis PPO.py (PPOAgent, SnakeEnv)
  - Action : argmax(logits) greedy

4 visualisations :
  1. Beeswarm plot  – impact global de chaque feature sur chaque action
  2. Waterfall plot – décomposition d'une décision par situation
  3. Force plot     – vue HTML interactive par situation
  4. Summary heatmap – matrice SHAP [feature × action] et [feature × situation]

Installation requise :
    pip install shap

Usage :
    python xai_shap_ppo.py --heatmap --episodes 5 --background 50   # test rapide
    python xai_shap_ppo.py --beeswarm --episodes 10
    python xai_shap_ppo.py --waterfall --episodes 10
    python xai_shap_ppo.py --force --episodes 10
    python xai_shap_ppo.py                                          # tout (12 épisodes)
"""

import argparse
import os
import math
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

from PPO import PPOAgent, SnakeEnv

OUT_DIR = "xai_shap_ppo"
os.makedirs(OUT_DIR, exist_ok=True)

def out(filename: str) -> str:
    return os.path.join(OUT_DIR, filename)


# ─── Constantes ────────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "Danger N",   "Danger NE",  "Danger E",   "Danger SE",
    "Danger S",   "Danger SW",  "Danger W",   "Danger NW",
    "Food N",     "Food NE",    "Food E",     "Food SE",
    "Food S",     "Food SW",    "Food W",     "Food NW",
    "Dir UP",     "Dir RIGHT",  "Dir DOWN",   "Dir LEFT",
    "Longueur",   "Urgence",
]
N_FEATURES = len(FEATURE_NAMES)   # 22

ACTION_NAMES  = ["UP ↑", "RIGHT →", "DOWN ↓", "LEFT ←"]
ACTION_COLORS = ["#4FC3F7", "#81C784", "#FFB74D", "#F06292"]

SITUATION_NAMES = [
    "Danger N", "Danger E", "Danger S", "Danger W",
    "Food alignée H", "Food alignée V", "Serpent long", "Neutre",
]
SITUATION_COLORS = [
    "#E74C3C", "#F39C12", "#F1C40F", "#2ECC71",
    "#3498DB", "#9B59B6", "#1ABC9C", "#95A5A6",
]

BG       = "#0D1117"
PANEL_BG = "#0D1B2A"
GRID_COL = "#1E3A5F"
TEXT_COL = "#CCDDEE"

CMAP_SHAP = LinearSegmentedColormap.from_list(
    "shap_div", ["#C0392B", "#E8A090", "#F5F5F5", "#90C8E8", "#1A5276"])
CMAP_ABS  = LinearSegmentedColormap.from_list(
    "shap_abs", ["#0D1B2A", "#154360", "#1F618D", "#F39C12", "#E74C3C"])


# ─── Wrapper : expose uniquement les logits ────────────────────────────────────
class ActorWrapper(nn.Module):
    """
    Encapsule ActorCritic pour n'exposer que les logits actor.
    Nécessaire pour SHAP DeepExplainer (sortie scalaire ou vecteur simple).
    """
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        logits, _ = self.net(x)
        return logits   # [batch, 4]


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


def _classify_situation(state: np.ndarray) -> int:
    DANGER_THR  = SnakeEnv.CELL / SnakeEnv._MAX_DIST
    thresh_long = 4.0 / (SnakeEnv.MAX_CELLS - 1)
    d_n, d_e, d_s, d_w = state[0], state[2], state[4], state[6]
    food_h = state[10] + state[14]
    food_v = state[8]  + state[12]
    if 0 < d_n <= DANGER_THR: return 0
    if 0 < d_e <= DANGER_THR: return 1
    if 0 < d_s <= DANGER_THR: return 2
    if 0 < d_w <= DANGER_THR: return 3
    if food_h > 0:             return 4
    if food_v > 0:             return 5
    if state[20] >= thresh_long: return 6
    return 7


def collect_states(agent: PPOAgent, env: SnakeEnv,
                   n_episodes: int = 12):
    all_states     = []
    all_actions    = []
    all_situations = []

    for ep in range(n_episodes):
        obs  = env.reset()
        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32,
                                 device=agent.device).unsqueeze(0)
            with torch.no_grad():
                logits, _ = agent.net(obs_t)
            action = int(logits.argmax(dim=1).item())

            all_states.append(obs.copy())
            all_actions.append(action)
            all_situations.append(_classify_situation(obs))

            obs, _, done, info = env.step(action)

        print(f"  [Collect] Épisode {ep+1}/{n_episodes} → score {info['score']}"
              f"  | total : {len(all_states)} steps")

    return (
        np.array(all_states,     dtype=np.float32),
        np.array(all_actions,    dtype=np.int32),
        np.array(all_situations, dtype=np.int32),
    )


def apply_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL_BG)
    if title:  ax.set_title(title,  color="white",  fontsize=11, fontweight="bold", pad=9)
    if xlabel: ax.set_xlabel(xlabel, color=TEXT_COL, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=TEXT_COL, fontsize=9)
    ax.tick_params(colors="#8899AA", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_COL)


# ─── Calcul des valeurs SHAP ───────────────────────────────────────────────────
def compute_shap_values(agent: PPOAgent,
                         states: np.ndarray,
                         background_size: int = 150):
    """
    DeepExplainer sur ActorWrapper (logits uniquement).
    Retourne (shap_values : list[np.ndarray [T,22]] × 4, expected_val [4]).
    """
    try:
        import shap
    except ImportError:
        raise ImportError("shap non installé.\n"
                          "Installez-le avec : pip install shap")

    wrapper = ActorWrapper(agent.net).to(agent.device)
    wrapper.eval()

    # Background : sous-ensemble aléatoire
    bg_size = min(background_size, len(states))
    bg_idx  = np.random.choice(len(states), bg_size, replace=False)
    bg_t    = torch.tensor(states[bg_idx], dtype=torch.float32,
                           device=agent.device)

    print(f"  [SHAP] Background : {bg_size} états | Total : {len(states)} états…")
    explainer   = shap.DeepExplainer(wrapper, bg_t)
    states_t    = torch.tensor(states, dtype=torch.float32, device=agent.device)
    shap_raw    = explainer.shap_values(states_t)
    expected    = explainer.expected_value

    if isinstance(expected, torch.Tensor):
        expected = expected.cpu().numpy()

    T  = len(states)
    F  = N_FEATURES   # 22
    A  = SnakeEnv.ACT_DIM   # 4

    # Normalisation vers liste de A arrays [T, F]
    if isinstance(shap_raw, (list, tuple)):
        shap_values = [
            (sv.cpu().numpy() if isinstance(sv, torch.Tensor) else np.array(sv))
            for sv in shap_raw
        ]
        shap_values = [sv if sv.shape == (T, F) else sv.T for sv in shap_values]
    else:
        arr = shap_raw.cpu().numpy() if isinstance(shap_raw, torch.Tensor) else np.array(shap_raw)
        print(f"  [SHAP] Raw shape : {arr.shape} – normalisation…")
        if arr.ndim == 3:
            if arr.shape == (T, F, A):
                shap_values = [arr[:, :, ai] for ai in range(A)]
            elif arr.shape == (A, T, F):
                shap_values = [arr[ai] for ai in range(A)]
            elif arr.shape == (T, A, F):
                shap_values = [arr[:, ai, :] for ai in range(A)]
            else:
                shap_values = [np.take(arr, ai, axis=2) for ai in range(A)]
        elif arr.ndim == 2 and arr.shape == (T, F):
            shap_values = [arr] * A
        else:
            raise ValueError(f"Shape SHAP non reconnue : {arr.shape}")

    print(f"  [SHAP] ✓ Shape par action : {shap_values[0].shape}")
    return shap_values, expected


# ─── Visualisation 1 : Beeswarm ───────────────────────────────────────────────
def plot_beeswarm(shap_values: list, states: np.ndarray):
    """
    Beeswarm plot pour les 4 actions.
    Axe X = valeur SHAP (impact sur logit).
    Couleur = valeur réelle de la feature (froid=faible, chaud=élevé).
    Features triées par |SHAP| moyen décroissant (les plus importantes en haut).
    """
    fig, axes = plt.subplots(1, 4, figsize=(26, 10), facecolor=BG)
    fig.suptitle(
        "SHAP Beeswarm – Impact de chaque feature sur les logits PPO\n"
        "Chaque point = un état  |  Axe X : valeur SHAP (+= pousse vers l'action)  |  "
        "Couleur : valeur de la feature (froid=faible, chaud=élevé)",
        fontsize=12, fontweight="bold", color="white", y=1.02
    )

    mean_abs_all = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    feat_order   = np.argsort(mean_abs_all)   # croissant → top en haut

    CMAP_FEAT = matplotlib.colormaps.get_cmap("coolwarm")

    for ai, ax in enumerate(axes):
        ax.set_facecolor(PANEL_BG)
        sv = shap_values[ai]
        T  = sv.shape[0]

        for rank, fi in enumerate(feat_order):
            shap_fi = sv[:, fi]
            feat_fi = states[:, fi]
            jitter  = np.random.uniform(-0.35, 0.35, size=T)
            f_min, f_max = feat_fi.min(), feat_fi.max()
            feat_norm = (feat_fi - f_min) / (f_max - f_min + 1e-8)
            ax.scatter(shap_fi, rank + jitter, c=feat_norm,
                       cmap=CMAP_FEAT, s=5, alpha=0.50,
                       edgecolors="none", vmin=0, vmax=1)

        ax.axvline(x=0, color="#AAAAAA", lw=1.0, ls="--", alpha=0.6)

        ax.set_yticks(range(N_FEATURES))
        ax.set_yticklabels([FEATURE_NAMES[fi] for fi in feat_order],
                           color=TEXT_COL, fontsize=8)

        # Séparateurs catégories sur axe Y
        for sep_val in [sum(1 for fi in feat_order if fi < 8) - 0.5,
                        sum(1 for fi in feat_order if fi < 16) - 0.5,
                        sum(1 for fi in feat_order if fi < 20) - 0.5]:
            ax.axhline(y=sep_val, color="#F39C12", lw=1.0, ls=":", alpha=0.6)

        apply_style(ax,
                    title=f"Action : {ACTION_NAMES[ai]}",
                    xlabel="Valeur SHAP (impact sur logit)")
        ax.set_title(f"Action : {ACTION_NAMES[ai]}",
                     color=ACTION_COLORS[ai], fontsize=12,
                     fontweight="bold", pad=9)
        ax.grid(axis="x", color=GRID_COL, lw=0.5, alpha=0.5)

    # Colorbar globale
    sm = ScalarMappable(cmap=CMAP_FEAT, norm=Normalize(0, 1))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.008, 0.65])
    cbar    = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Valeur feature (normalisée)", color=TEXT_COL,
                   fontsize=9, rotation=270, labelpad=14)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COL, labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COL)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(["Faible", "Moyen", "Élevé"])

    plt.subplots_adjust(right=0.90, wspace=0.45)
    plt.savefig(out("xai_shap_beeswarm.png"), dpi=150,
                bbox_inches="tight", facecolor=BG)
    print(f"[XAI] Sauvegarde → {out('xai_shap_beeswarm.png')}")
    plt.show()


# ─── Visualisation 2 : Waterfall ──────────────────────────────────────────────
def plot_waterfall(shap_values: list, states: np.ndarray,
                   situations: np.ndarray, expected: np.ndarray):
    """
    Pour chaque situation, sélectionne l'état médian (le plus représentatif)
    et trace un waterfall de contributions SHAP cumulées vers le logit final.
    """
    from collections import Counter

    n_sit  = len(SITUATION_NAMES)
    n_cols = 4
    n_rows = math.ceil(n_sit / n_cols)

    fig = plt.figure(figsize=(24, 7 * n_rows), facecolor=BG)
    fig.suptitle(
        "SHAP Waterfall – Décomposition d'une décision par situation – PPO\n"
        "Chaque barre = contribution d'une feature au logit final  |  "
        "Départ = E[f(x)]  →  Arrivée = logit prédit",
        fontsize=12, fontweight="bold", color="white", y=1.01
    )
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                           wspace=0.50, hspace=0.70)

    for si in range(n_sit):
        row = si // n_cols
        col = si  % n_cols
        ax  = fig.add_subplot(gs[row, col])
        ax.set_facecolor(PANEL_BG)

        mask = situations == si
        if mask.sum() == 0:
            ax.set_visible(False)
            continue

        indices = np.where(mask)[0]

        # Action dominante dans cette situation
        action_counts = Counter()
        for idx in indices:
            sv_all = np.array([shap_values[ai][idx] for ai in range(SnakeEnv.ACT_DIM)])
            action_counts[int(sv_all.sum(axis=1).argmax())] += 1
        dominant_action = action_counts.most_common(1)[0][0]

        sv_action  = shap_values[dominant_action]
        total_shap = np.abs(sv_action[indices]).sum(axis=1)
        median_val = np.median(total_shap)
        rep_idx    = indices[np.argmin(np.abs(total_shap - median_val))]

        shap_rep  = sv_action[rep_idx]    # [22]
        state_rep = states[rep_idx]        # [22]
        base_val  = (float(expected[dominant_action])
                     if hasattr(expected, '__len__') else float(expected))

        # Tri par |SHAP| décroissant
        order      = np.argsort(np.abs(shap_rep))[::-1]
        shap_ord   = shap_rep[order]
        feat_vals  = state_rep[order]

        # Positions cumulatives
        cumulative    = np.zeros(N_FEATURES + 1)
        cumulative[0] = base_val
        for k, s in enumerate(shap_ord):
            cumulative[k + 1] = cumulative[k] + s
        final_val = cumulative[-1]

        bar_bottoms = cumulative[:-1].copy()
        bar_heights = shap_ord.copy()
        for k in range(N_FEATURES):
            if shap_ord[k] < 0:
                bar_bottoms[k] = cumulative[k + 1]
                bar_heights[k] = -shap_ord[k]

        colors_wf = ["#2E86C1" if s >= 0 else "#C0392B" for s in shap_ord]
        ax.barh(range(N_FEATURES), bar_heights, left=bar_bottoms,
                color=colors_wf, edgecolor="#0D1117", height=0.68, alpha=0.88)

        for k, (b, h, s) in enumerate(zip(bar_bottoms, bar_heights, shap_ord)):
            x_txt = b + h + (0.02 if s >= 0 else -0.02)
            ax.text(x_txt, k, f"{s:+.3f}", va="center",
                    ha="left" if s >= 0 else "right",
                    fontsize=6, color="#AADDFF" if s >= 0 else "#FFAAAA")

        ax.axvline(x=base_val,  color="#F39C12", lw=1.2, ls="--", alpha=0.8)
        ax.axvline(x=final_val, color="#2ECC71", lw=1.4, ls="-",  alpha=0.8)

        ax.set_yticks(range(N_FEATURES))
        ax.set_yticklabels(
            [f"{FEATURE_NAMES[order[k]]}  [{feat_vals[k]:.2f}]"
             for k in range(N_FEATURES)],
            fontsize=6.5, color=TEXT_COL
        )
        ax.set_title(f"{SITUATION_NAMES[si]}  →  {ACTION_NAMES[dominant_action]}",
                     color=SITUATION_COLORS[si], fontsize=10,
                     fontweight="bold", pad=7)
        ax.set_xlabel("Logit (contributions cumulées)", color=TEXT_COL, fontsize=8)
        ax.tick_params(colors="#8899AA", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID_COL)
        ax.grid(axis="x", color=GRID_COL, lw=0.5, alpha=0.4)

        pos_p = mpatches.Patch(color="#2E86C1", label=f"+ (E[f]={base_val:.2f})")
        neg_p = mpatches.Patch(color="#C0392B", label=f"– (f(x)={final_val:.2f})")
        ax.legend(handles=[pos_p, neg_p], fontsize=6.5, facecolor="#0D1117",
                  edgecolor="#444", labelcolor="white", loc="lower right")

    plt.savefig(out("xai_shap_waterfall.png"), dpi=150,
                bbox_inches="tight", facecolor=BG)
    print(f"[XAI] Sauvegarde → {out('xai_shap_waterfall.png')}")
    plt.show()


# ─── Visualisation 3 : Force plot HTML ────────────────────────────────────────
def plot_force(shap_values: list, states: np.ndarray,
               situations: np.ndarray, expected: np.ndarray):
    """
    Génère des force plots HTML (un par situation + un global).
    """
    try:
        import shap
    except ImportError:
        print("[SKIP] shap non installé.")
        return

    from collections import Counter
    shap.initjs()

    for si in range(len(SITUATION_NAMES)):
        mask = situations == si
        if mask.sum() == 0:
            continue
        indices = np.where(mask)[0][:50]

        action_counts = Counter()
        for idx in indices:
            sv_all = np.array([shap_values[ai][idx] for ai in range(SnakeEnv.ACT_DIM)])
            action_counts[int(sv_all.sum(axis=1).argmax())] += 1
        dom_a    = action_counts.most_common(1)[0][0]
        sv_sit   = shap_values[dom_a][indices]
        st_sit   = states[indices]
        base_val = (float(expected[dom_a])
                    if hasattr(expected, '__len__') else float(expected))

        html_path = out(
            f"xai_force_sit{si}_{SITUATION_NAMES[si].replace(' ', '_')}.html"
        )
        try:
            fp = shap.force_plot(float(base_val), sv_sit, st_sit,
                                 feature_names=FEATURE_NAMES,
                                 show=False, matplotlib=False)
            shap.save_html(html_path, fp)
            print(f"[XAI] Sauvegarde → {html_path}")
        except Exception as e:
            print(f"[WARN] Force plot sit.{si} : {e}")

    # Force plot global
    from collections import Counter
    all_best = []
    for i in range(len(situations)):
        sv_all = np.array([shap_values[ai][i] for ai in range(SnakeEnv.ACT_DIM)])
        all_best.append(int(sv_all.sum(axis=1).argmax()))
    dom_g    = Counter(all_best).most_common(1)[0][0]
    sv_g     = shap_values[dom_g]
    base_g   = (float(expected[dom_g])
                if hasattr(expected, '__len__') else float(expected))
    MAX_HTML = 500
    idx_html = np.linspace(0, len(situations) - 1,
                           min(MAX_HTML, len(situations)), dtype=int)
    try:
        fp_g = shap.force_plot(base_g, sv_g[idx_html], states[idx_html],
                               feature_names=FEATURE_NAMES,
                               show=False, matplotlib=False)
        shap.save_html(out("xai_force_global.html"), fp_g)
        print(f"[XAI] Sauvegarde → {out('xai_force_global.html')}")
    except Exception as e:
        print(f"[WARN] Force plot global : {e}")


# ─── Visualisation 4 : Summary heatmap ────────────────────────────────────────
def plot_summary_heatmap(shap_values: list, states: np.ndarray,
                          situations: np.ndarray):
    """
    4 sous-figures :
      A) |SHAP| moyen par feature × action (importance absolue)
      B) SHAP signé moyen par feature × action (direction d'influence)
      C) Barplot importance globale (toutes actions)
      D) |SHAP| moyen par feature × situation
    """
    # Matrices de base
    mean_abs_matrix  = np.zeros((N_FEATURES, SnakeEnv.ACT_DIM))
    mean_sign_matrix = np.zeros((N_FEATURES, SnakeEnv.ACT_DIM))
    for ai in range(SnakeEnv.ACT_DIM):
        mean_abs_matrix[:, ai]  = np.abs(shap_values[ai]).mean(axis=0)
        mean_sign_matrix[:, ai] = shap_values[ai].mean(axis=0)

    global_importance = mean_abs_matrix.mean(axis=1)
    feat_order        = np.argsort(global_importance)   # croissant

    mean_sit_matrix = np.zeros((N_FEATURES, len(SITUATION_NAMES)))
    for si in range(len(SITUATION_NAMES)):
        mask = situations == si
        if mask.sum() == 0:
            continue
        for ai in range(SnakeEnv.ACT_DIM):
            mean_sit_matrix[:, si] += np.abs(shap_values[ai][mask]).mean(axis=0)
        mean_sit_matrix[:, si] /= SnakeEnv.ACT_DIM

    fig = plt.figure(figsize=(24, 15), facecolor=BG)
    fig.suptitle(
        "SHAP Summary – PPO Snake – Vue globale de l'importance des 22 features\n"
        "Calculé sur l'ensemble des états collectés",
        fontsize=14, fontweight="bold", color="white"
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.40, hspace=0.55)

    # A) |SHAP| feature × action
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_facecolor(PANEL_BG)
    data_a = mean_abs_matrix[feat_order, :]
    vmax_a = np.percentile(data_a, 97)
    im_a   = ax_a.imshow(data_a, cmap=CMAP_ABS, vmin=0, vmax=max(vmax_a, 1e-6),
                          aspect="auto", interpolation="nearest")
    for fi_r, fi in enumerate(feat_order):
        for ai in range(SnakeEnv.ACT_DIM):
            v = mean_abs_matrix[fi, ai]
            c = "white" if v > vmax_a * 0.5 else TEXT_COL
            ax_a.text(ai, fi_r, f"{v:.3f}", ha="center", va="center",
                      color=c, fontsize=7.5)
    ax_a.set_xticks(range(SnakeEnv.ACT_DIM))
    ax_a.set_xticklabels(ACTION_NAMES, color=TEXT_COL, fontsize=9)
    ax_a.set_yticks(range(N_FEATURES))
    ax_a.set_yticklabels([FEATURE_NAMES[i] for i in feat_order],
                          color=TEXT_COL, fontsize=8)
    # Séparateurs catégories
    for sep in [sum(1 for i in feat_order if i < 8) - 0.5,
                sum(1 for i in feat_order if i < 16) - 0.5,
                sum(1 for i in feat_order if i < 20) - 0.5]:
        ax_a.axhline(y=sep, color="#F39C12", lw=1.2, ls="--", alpha=0.7)
    cbar_a = plt.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.04)
    cbar_a.set_label("|SHAP| moyen", color=TEXT_COL, fontsize=8)
    cbar_a.ax.yaxis.set_tick_params(color=TEXT_COL, labelsize=7)
    plt.setp(cbar_a.ax.yaxis.get_ticklabels(), color=TEXT_COL)
    apply_style(ax_a, title="|SHAP| moyen par feature × action\n"
                             "(importance absolue – plus clair = plus impactant)")

    # B) SHAP signé feature × action
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_facecolor(PANEL_BG)
    data_b = mean_sign_matrix[feat_order, :]
    vabs_b = np.abs(data_b).max()
    norm_b = TwoSlopeNorm(vcenter=0, vmin=-vabs_b, vmax=vabs_b)
    im_b   = ax_b.imshow(data_b, cmap=CMAP_SHAP, norm=norm_b,
                          aspect="auto", interpolation="nearest")
    for fi_r, fi in enumerate(feat_order):
        for ai in range(SnakeEnv.ACT_DIM):
            v   = mean_sign_matrix[fi, ai]
            col = "white" if abs(v) > vabs_b * 0.4 else TEXT_COL
            ax_b.text(ai, fi_r, f"{v:+.3f}", ha="center", va="center",
                      color=col, fontsize=7.5)
    ax_b.set_xticks(range(SnakeEnv.ACT_DIM))
    ax_b.set_xticklabels(ACTION_NAMES, color=TEXT_COL, fontsize=9)
    ax_b.set_yticks(range(N_FEATURES))
    ax_b.set_yticklabels([FEATURE_NAMES[i] for i in feat_order],
                          color=TEXT_COL, fontsize=8)
    for sep in [sum(1 for i in feat_order if i < 8) - 0.5,
                sum(1 for i in feat_order if i < 16) - 0.5,
                sum(1 for i in feat_order if i < 20) - 0.5]:
        ax_b.axhline(y=sep, color="#F39C12", lw=1.2, ls="--", alpha=0.7)
    cbar_b = plt.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04)
    cbar_b.set_label("SHAP signé moyen", color=TEXT_COL, fontsize=8)
    cbar_b.ax.yaxis.set_tick_params(color=TEXT_COL, labelsize=7)
    plt.setp(cbar_b.ax.yaxis.get_ticklabels(), color=TEXT_COL)
    apply_style(ax_b, title="SHAP signé moyen par feature × action\n"
                             "(bleu = impact +, rouge = impact –)")

    # C) Barplot importance globale
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_facecolor(PANEL_BG)
    gi_sorted = global_importance[feat_order]
    norm_c    = Normalize(vmin=gi_sorted.min(), vmax=gi_sorted.max())
    colors_c  = [CMAP_ABS(norm_c(v)) for v in gi_sorted]
    ax_c.barh(range(N_FEATURES), gi_sorted, color=colors_c,
              edgecolor="#0D1117", height=0.72)
    for k in range(N_FEATURES):
        ax_c.axhspan(k - 0.5, k + 0.5,
                     color="#0F2233" if k % 2 == 0 else PANEL_BG,
                     alpha=0.4, zorder=0)
    for k, v in enumerate(gi_sorted):
        ax_c.text(v + 0.0002, k, f"{v:.4f}", va="center",
                  color=TEXT_COL, fontsize=7.5)
    ax_c.set_yticks(range(N_FEATURES))
    ax_c.set_yticklabels([FEATURE_NAMES[i] for i in feat_order],
                          color=TEXT_COL, fontsize=8.5)
    for sep in [sum(1 for i in feat_order if i < 8) - 0.5,
                sum(1 for i in feat_order if i < 16) - 0.5,
                sum(1 for i in feat_order if i < 20) - 0.5]:
        ax_c.axhline(y=sep, color="#F39C12", lw=1.2, ls="--", alpha=0.7)
    apply_style(ax_c,
                title="Importance SHAP globale (toutes actions)\n"
                      "Rang ↑ = feature la plus influente",
                xlabel="|SHAP| moyen (toutes actions)")
    ax_c.grid(axis="x", color=GRID_COL, lw=0.5, alpha=0.5)

    # D) |SHAP| feature × situation
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_facecolor(PANEL_BG)
    data_d = mean_sit_matrix[feat_order, :]
    vmax_d = np.percentile(data_d, 97)
    im_d   = ax_d.imshow(data_d, cmap=CMAP_ABS, vmin=0, vmax=max(vmax_d, 1e-6),
                          aspect="auto", interpolation="nearest")
    ax_d.set_xticks(range(len(SITUATION_NAMES)))
    ax_d.set_xticklabels([s.replace(" ", "\n") for s in SITUATION_NAMES],
                          color=TEXT_COL, fontsize=7.5)
    ax_d.set_yticks(range(N_FEATURES))
    ax_d.set_yticklabels([FEATURE_NAMES[i] for i in feat_order],
                          color=TEXT_COL, fontsize=8)
    for sep in [sum(1 for i in feat_order if i < 8) - 0.5,
                sum(1 for i in feat_order if i < 16) - 0.5,
                sum(1 for i in feat_order if i < 20) - 0.5]:
        ax_d.axhline(y=sep, color="#F39C12", lw=1.2, ls="--", alpha=0.7)
    for si, col in enumerate(SITUATION_COLORS):
        ax_d.axvline(x=si - 0.5, color=col, lw=0.6, alpha=0.4)
    cbar_d = plt.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.04)
    cbar_d.set_label("|SHAP| moyen", color=TEXT_COL, fontsize=8)
    cbar_d.ax.yaxis.set_tick_params(color=TEXT_COL, labelsize=7)
    plt.setp(cbar_d.ax.yaxis.get_ticklabels(), color=TEXT_COL)
    apply_style(ax_d, title="|SHAP| moyen par feature × situation\n"
                             "(quelle feature est cruciale dans quelle situation ?)")

    plt.savefig(out("xai_shap_heatmap.png"), dpi=150,
                bbox_inches="tight", facecolor=BG)
    print(f"[XAI] Sauvegarde → {out('xai_shap_heatmap.png')}")
    plt.show()


# ─── Point d'entrée ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="XAI – SHAP PPO Snake")
    parser.add_argument("--beeswarm",  action="store_true")
    parser.add_argument("--waterfall", action="store_true")
    parser.add_argument("--force",     action="store_true")
    parser.add_argument("--heatmap",   action="store_true")
    parser.add_argument("--model",      type=str, default="model_best.pth")
    parser.add_argument("--episodes",   type=int, default=12)
    parser.add_argument("--background", type=int, default=150,
                        help="Taille du background SHAP (défaut : 150)")
    args = parser.parse_args()

    run_all = not (args.beeswarm or args.waterfall or args.force or args.heatmap)

    try:
        import shap
        print(f"[XAI] shap version : {shap.__version__}")
    except ImportError:
        print("[ERREUR] shap non installé.\n         pip install shap")
        return

    agent = load_agent(args.model)
    env   = SnakeEnv(render=False)

    print(f"\n[XAI] Collecte sur {args.episodes} épisode(s)…")
    states, actions, situations = collect_states(agent, env, args.episodes)
    env.close()
    print(f"[XAI] {len(states)} états collectés.\n")

    print("[XAI] Calcul des valeurs SHAP (DeepExplainer)…")
    shap_values, expected = compute_shap_values(
        agent, states, background_size=args.background
    )

    if run_all or args.beeswarm:
        print("\n[XAI] ── Beeswarm ──")
        plot_beeswarm(shap_values, states)

    if run_all or args.waterfall:
        print("\n[XAI] ── Waterfall ──")
        plot_waterfall(shap_values, states, situations, expected)

    if run_all or args.heatmap:
        print("\n[XAI] ── Summary heatmap ──")
        plot_summary_heatmap(shap_values, states, situations)

    if run_all or args.force:
        print("\n[XAI] ── Force plots HTML ──")
        plot_force(shap_values, states, situations, expected)

    print(f"\n[XAI] Analyse SHAP terminée. Fichiers dans : {OUT_DIR}/")


if __name__ == "__main__":
    main()

# ── Tests recommandés ──────────────────────────────────────────────────────────
# Prérequis : pip install shap
#
# Test rapide (~1-2min) :
#   python xai_shap_ppo.py --heatmap --episodes 5 --background 50
#   → Vérifie : xai_shap_ppo/xai_shap_heatmap.png
#     ✓ 4 sous-figures (|SHAP|×action, signé×action, barplot, ×situation)
#     ✓ 22 features sur l'axe Y
#     ✓ Séparateurs orange entre catégories (danger / food / direction / contexte)
#
# Test beeswarm (~2-3min) :
#   python xai_shap_ppo.py --beeswarm --episodes 10 --background 100
#   → Vérifie : xai_shap_ppo/xai_shap_beeswarm.png
#     ✓ 4 subplots (un par action)
#     ✓ Nuage de points colorés par valeur de feature
#     ✓ Features en haut = plus d'impact
#
# Test waterfall (~2min) :
#   python xai_shap_ppo.py --waterfall --episodes 10
#   → Vérifie : xai_shap_ppo/xai_shap_waterfall.png
#     ✓ 8 waterfall (une par situation)
#     ✓ Barres bleues = contributions positives, rouges = négatives
#
# Test force plots (génère des HTML interactifs) :
#   python xai_shap_ppo.py --force --episodes 10
#   → Vérifie : xai_shap_ppo/xai_force_sit*.html  (ouvrir dans navigateur)
