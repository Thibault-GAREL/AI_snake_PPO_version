"""
xai_activations_ppo.py – Analyse XAI : Activations internes du PPO Snake
=========================================================================
3 analyses :
  1. Distribution des activations par couche
       → histogrammes, taux de saturation Tanh (|act| > 0.99) et quasi-nuls (|act| < 0.05)
  2. Neurones spécialisés
       → quels neurones s'activent uniquement dans des situations précises
  3. t-SNE / UMAP des activations
       → projection 2D des états vus → clusters de situations similaires

Architecture ActorCritic (PPO.py) :
    shared[2]      Tanh  ← couche partagée 1 (256n)
    shared[5]      Tanh  ← couche partagée 2 (256n)
    actor_head[1]  Tanh  ← actor head       (128n)
    critic_head[1] Tanh  ← critic head      (128n)

Usage :
    python xai_activations_ppo.py --distribution --episodes 5   # test rapide
    python xai_activations_ppo.py --specialization --episodes 10
    python xai_activations_ppo.py --tsne --episodes 15
    python xai_activations_ppo.py --umap --episodes 15
    python xai_activations_ppo.py                               # tout (10 épisodes)
"""

import argparse
import os
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import torch

warnings.filterwarnings("ignore")

from PPO import PPOAgent, SnakeEnv

OUT_DIR = "xai_activations_ppo"
os.makedirs(OUT_DIR, exist_ok=True)

def out(filename: str) -> str:
    return os.path.join(OUT_DIR, filename)


# ─── Constantes ────────────────────────────────────────────────────────────────
LAYER_HOOKS = {
    "Partagée 1\n(Tanh, 256n)":  ("shared",      2),
    "Partagée 2\n(Tanh, 256n)":  ("shared",      5),
    "Actor head\n(Tanh, 128n)":  ("actor_head",  1),
    "Critic head\n(Tanh, 128n)": ("critic_head", 1),
}
LAYER_SIZES = {
    "Partagée 1\n(Tanh, 256n)":  256,
    "Partagée 2\n(Tanh, 256n)":  256,
    "Actor head\n(Tanh, 128n)":  128,
    "Critic head\n(Tanh, 128n)": 128,
}
LAYER_KEYS = list(LAYER_HOOKS.keys())

ACTION_NAMES  = ["UP ↑", "RIGHT →", "DOWN ↓", "LEFT ←"]
ACTION_COLORS = ["#4FC3F7", "#81C784", "#FFB74D", "#F06292"]

SITUATION_NAMES = [
    "Danger N", "Danger E", "Danger S", "Danger W",
    "Food alignée H", "Food alignée V", "Serpent long\n(≥5)", "Neutre",
]

BG       = "#0D1117"
PANEL_BG = "#0D1B2A"
GRID_COL = "#1E3A5F"
TEXT_COL = "#CCDDEE"

CMAP_SAT  = LinearSegmentedColormap.from_list("sat",  ["#2ECC71", "#F39C12", "#E74C3C"])
CMAP_SPEC = LinearSegmentedColormap.from_list("spec", ["#0D1B2A", "#154360", "#1F618D", "#D4AC0D", "#E74C3C"])


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


# ─── Classifieur de situation (state PPO 22-dim) ───────────────────────────────
def _classify_situation(state: np.ndarray) -> int:
    """
    Classe l'état courant en 8 situations.
    Features PPO :
      [0:8]  dangers normalisés (N NE E SE S SW W NW)
      [8:16] nourriture normalisée
      [16:20] one-hot direction
      [20] longueur norm, [21] urgence
    """
    DANGER_THR  = SnakeEnv.CELL / SnakeEnv._MAX_DIST   # 1 case de distance
    thresh_long = 4.0 / (SnakeEnv.MAX_CELLS - 1)       # longueur ≥ 5

    d_n, d_e, d_s, d_w = state[0], state[2], state[4], state[6]
    food_h = state[10] + state[14]   # food E + food W  (alignée horizontalement)
    food_v = state[8]  + state[12]   # food N + food S  (alignée verticalement)

    if 0 < d_n <= DANGER_THR:      return 0
    if 0 < d_e <= DANGER_THR:      return 1
    if 0 < d_s <= DANGER_THR:      return 2
    if 0 < d_w <= DANGER_THR:      return 3
    if food_h > 0:                  return 4
    if food_v > 0:                  return 5
    if state[20] >= thresh_long:    return 6
    return 7


# ─── Collecteur d'activations via forward hooks ────────────────────────────────
class ActivationCollector:
    def __init__(self, agent: PPOAgent):
        self.agent  = agent
        self.data   = {k: [] for k in LAYER_KEYS}
        self._hooks = []
        self._register()

    def _register(self):
        for layer_name, (block_name, idx) in LAYER_HOOKS.items():
            submodule = getattr(self.agent.net, block_name)
            def make_hook(name):
                def hook(module, inp, out):
                    self.data[name].append(out.detach().cpu().numpy())
                return hook
            h = submodule[idx].register_forward_hook(make_hook(layer_name))
            self._hooks.append(h)

    def remove(self):
        for h in self._hooks:
            h.remove()

    def clear(self):
        self.data = {k: [] for k in LAYER_KEYS}

    def get_arrays(self) -> dict:
        return {k: np.vstack(v) for k, v in self.data.items() if len(v) > 0}


def collect_episodes(agent, env, collector, n_episodes: int = 10):
    states_log     = []
    actions_log    = []
    situations_log = []
    scores_log     = []
    collector.clear()

    for ep in range(n_episodes):
        obs  = env.reset()
        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32,
                                 device=agent.device).unsqueeze(0)
            with torch.no_grad():
                logits, _ = agent.net(obs_t)
            action = int(logits.argmax(dim=1).item())

            states_log.append(obs.copy())
            actions_log.append(action)
            situations_log.append(_classify_situation(obs))
            scores_log.append(env._score)

            obs, _, done, info = env.step(action)

        print(f"  [Collect] Épisode {ep+1}/{n_episodes} → score {info['score']}"
              f"  ({len(states_log)} steps total)")

    return states_log, actions_log, situations_log, scores_log


def apply_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL_BG)
    if title:  ax.set_title(title,  color="white",  fontsize=11, fontweight="bold", pad=8)
    if xlabel: ax.set_xlabel(xlabel, color=TEXT_COL, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color=TEXT_COL, fontsize=8)
    ax.tick_params(colors="#8899AA", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_COL)


# ─── Analyse 1 : Distribution des activations ─────────────────────────────────
def plot_distribution(act_arrays: dict):
    """
    Par couche :
      - Histogramme des valeurs Tanh ∈ [-1, 1]
      - Taux de neurones saturés (|act| > 0.99) et quasi-nuls (|act| < 0.05)
      - Heatmap temporelle (variance des neurones)
    """
    n_layers = len(LAYER_KEYS)
    fig = plt.figure(figsize=(22, 6 * n_layers), facecolor=BG)
    fig.suptitle(
        "Distribution des activations par couche – PPO ActorCritic\n"
        "Tanh ∈ [-1, 1]  |  Saturés = |act| > 0.99  |  Quasi-nuls = |act| < 0.05",
        fontsize=14, fontweight="bold", color="white", y=1.01
    )
    gs = gridspec.GridSpec(n_layers, 3, figure=fig,
                           wspace=0.38, hspace=0.55, width_ratios=[1.4, 1, 2])

    SAT_THR   = 0.99
    NEAR_ZERO = 0.05

    for row, layer_name in enumerate(LAYER_KEYS):
        acts = act_arrays[layer_name]   # [T, N]
        T, N = acts.shape

        frac_sat  = (np.abs(acts) > SAT_THR).mean(axis=0)    # [N]
        frac_zero = (np.abs(acts) < NEAR_ZERO).mean(axis=0)  # [N]
        n_sat     = (frac_sat  > 0.50).sum()
        n_near0   = (frac_zero > 0.50).sum()

        # Col 0 : histogramme
        ax0 = fig.add_subplot(gs[row, 0])
        ax0.hist(acts.flatten(), bins=80, color="#1F618D", alpha=0.65,
                 label="Toutes", edgecolor="none")
        ax0.set_yscale("log")
        for sign in [+1, -1]:
            ax0.axvline(x=sign * SAT_THR, color="#E74C3C",
                        ls="--", lw=1.3, alpha=0.9,
                        label=f"±{SAT_THR}" if sign == 1 else "")
        apply_style(ax0,
                    title=f"{layer_name.strip()} – Distribution",
                    xlabel="Valeur d'activation (Tanh ∈ [-1, 1])",
                    ylabel="Fréquence (log)")
        ax0.legend(fontsize=7, facecolor="#0D1117", edgecolor="#444",
                   labelcolor="white")
        ax0.text(0.97, 0.97,
                 f"min  = {acts.min():.3f}\n"
                 f"max  = {acts.max():.3f}\n"
                 f"mean = {acts.mean():.3f}\n"
                 f"std  = {acts.std():.3f}\n"
                 f"saturés  = {n_sat}/{N} ({100*n_sat/N:.1f}%)\n"
                 f"≈ zéro   = {n_near0}/{N} ({100*n_near0/N:.1f}%)",
                 transform=ax0.transAxes, va="top", ha="right",
                 color=TEXT_COL, fontsize=7,
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#0D1B2A",
                           edgecolor=GRID_COL, alpha=0.9))

        # Col 1 : saturation par neurone
        ax1 = fig.add_subplot(gs[row, 1])
        order    = np.argsort(frac_sat)[::-1]
        top_n    = min(40, N)
        bar_cols = [CMAP_SAT(frac_sat[order[i]]) for i in range(top_n)]
        ax1.barh(range(top_n), frac_sat[order[:top_n]],
                 color=bar_cols, edgecolor="none", height=0.85)
        ax1.axvline(x=0.50, color="#E74C3C", lw=1.2, ls="--", alpha=0.9,
                    label="Seuil 50%")
        ax1.set_yticks([])
        ax1.set_xlim(0, 1.05)
        apply_style(ax1,
                    title=f"Saturation (top {top_n})\n"
                          f"{n_sat}/{N} saturés ({100*n_sat/N:.1f}%)",
                    xlabel="Fraction de steps |act| > 0.99")
        ax1.legend(fontsize=7, facecolor="#0D1117", edgecolor="#444",
                   labelcolor="white")
        sm = ScalarMappable(cmap=CMAP_SAT, norm=Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax1, fraction=0.046, pad=0.04)
        cbar.set_label("Frac. sat.", color=TEXT_COL, fontsize=7)
        cbar.ax.yaxis.set_tick_params(color=TEXT_COL, labelsize=6)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COL)

        # Col 2 : heatmap temporelle
        ax2 = fig.add_subplot(gs[row, 2])
        t_show    = min(200, T)
        n_show    = min(80, N)
        var_order = np.argsort(acts.var(axis=0))[::-1][:n_show]
        heat      = acts[:t_show, var_order].T
        im = ax2.imshow(heat, cmap=CMAP_SPEC,
                        vmin=-1.0, vmax=1.0,
                        aspect="auto", interpolation="nearest")
        ax2.set_xlabel("Step (temps)", color=TEXT_COL, fontsize=8)
        ax2.set_ylabel(f"Neurone (top {n_show} variance)", color=TEXT_COL, fontsize=8)
        ax2.set_title(
            f"Activité temporelle – {n_show} neurones × {t_show} steps\n"
            "(triés par variance décroissante – foncé = inactif, chaud = actif fort)",
            color="white", fontsize=10, fontweight="bold", pad=8
        )
        ax2.tick_params(colors="#8899AA", labelsize=7)
        for sp in ax2.spines.values():
            sp.set_edgecolor(GRID_COL)
        cbar2 = plt.colorbar(im, ax=ax2, fraction=0.025, pad=0.02)
        cbar2.set_label("Activation Tanh", color=TEXT_COL, fontsize=7)
        cbar2.ax.yaxis.set_tick_params(color=TEXT_COL, labelsize=6)
        plt.setp(cbar2.ax.yaxis.get_ticklabels(), color=TEXT_COL)

    plt.savefig(out("xai_distribution.png"), dpi=150,
                bbox_inches="tight", facecolor=BG)
    print(f"[XAI] Sauvegarde → {out('xai_distribution.png')}")
    plt.show()


# ─── Analyse 2 : Neurones spécialisés ─────────────────────────────────────────
def compute_specialization(act_arrays: dict, situations: list) -> dict:
    situations_arr = np.array(situations)
    result = {}
    for layer_name, acts in act_arrays.items():
        N     = acts.shape[1]
        S     = len(SITUATION_NAMES)
        means = np.zeros((N, S))
        for si in range(S):
            mask = situations_arr == si
            if mask.sum() > 0:
                means[:, si] = acts[mask].mean(axis=0)
        result[layer_name] = means
    return result


def plot_specialization(spec_data: dict, situations: list, act_arrays: dict):
    situations_arr = np.array(situations)
    sit_counts = [(situations_arr == si).sum() for si in range(len(SITUATION_NAMES))]

    fig = plt.figure(figsize=(24, 7 * len(LAYER_KEYS)), facecolor=BG)
    fig.suptitle(
        "Neurones spécialisés – PPO ActorCritic\n"
        "Score = max_situation(act_moy) – mean_situations(act_moy)  "
        "| Score élevé = neurone très sélectif",
        fontsize=13, fontweight="bold", color="white", y=1.005
    )
    gs = gridspec.GridSpec(len(LAYER_KEYS), 3, figure=fig,
                           wspace=0.42, hspace=0.60, width_ratios=[1.2, 2, 1.8])

    for row, layer_name in enumerate(LAYER_KEYS):
        means      = spec_data[layer_name]   # [N, S]
        N          = means.shape[0]
        spec_score = means.max(axis=1) - means.mean(axis=1)
        top_idx    = np.argsort(spec_score)[::-1]

        # Col 0 : distribution des scores de spécialisation
        ax0 = fig.add_subplot(gs[row, 0])
        ax0.set_facecolor(PANEL_BG)
        ax0.hist(spec_score, bins=50, color="#1F618D", alpha=0.7, edgecolor="none")
        ax0.axvline(x=np.percentile(spec_score, 90), color="#E74C3C",
                    lw=1.5, ls="--", label="90e percentile")
        ax0.axvline(x=np.percentile(spec_score, 50), color="#F39C12",
                    lw=1.0, ls=":", label="médiane")
        apply_style(ax0,
                    title=f"{layer_name.strip()} – Score de spécialisation",
                    xlabel="max – mean des activations par situation",
                    ylabel="Nombre de neurones")
        ax0.legend(fontsize=7, facecolor="#0D1117", edgecolor="#444", labelcolor="white")
        ax0.text(0.97, 0.97,
                 f"Top neurone : #{top_idx[0]}\n"
                 f"Score max   : {spec_score[top_idx[0]]:.3f}\n"
                 f"Situation   : {SITUATION_NAMES[means[top_idx[0]].argmax()].replace(chr(10), ' ')}",
                 transform=ax0.transAxes, va="top", ha="right",
                 color="#FFD700", fontsize=7.5,
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#0D1B2A",
                           edgecolor="#F39C12", alpha=0.9))

        # Col 1 : heatmap [situation × top-40 neurones]
        ax1 = fig.add_subplot(gs[row, 1])
        ax1.set_facecolor(PANEL_BG)
        top40  = top_idx[:40]
        heat   = means[top40, :].T
        vmax_h = np.percentile(means, 95)
        im = ax1.imshow(heat, cmap=CMAP_SPEC,
                        vmin=0, vmax=max(vmax_h, 1e-6),
                        aspect="auto", interpolation="nearest")
        ax1.set_yticks(range(len(SITUATION_NAMES)))
        ax1.set_yticklabels(
            [f"{n.replace(chr(10), ' ')}  (n={sit_counts[i]})"
             for i, n in enumerate(SITUATION_NAMES)],
            color=TEXT_COL, fontsize=8
        )
        ax1.set_xlabel("Neurone (top 40 spécialisés)", color=TEXT_COL, fontsize=8)
        ax1.set_title(
            "Activation moyenne par situation × neurone\n"
            "(triés par score de spécialisation décroissant)",
            color="white", fontsize=10, fontweight="bold", pad=8
        )
        ax1.tick_params(colors="#8899AA", labelsize=8)
        for sp in ax1.spines.values():
            sp.set_edgecolor(GRID_COL)
        cbar1 = plt.colorbar(im, ax=ax1, fraction=0.03, pad=0.03)
        cbar1.set_label("Activation moy.", color=TEXT_COL, fontsize=7)
        cbar1.ax.yaxis.set_tick_params(color=TEXT_COL, labelsize=6)
        plt.setp(cbar1.ax.yaxis.get_ticklabels(), color=TEXT_COL)

        # Col 2 : profil des top-5 neurones
        ax2 = fig.add_subplot(gs[row, 2])
        ax2.set_facecolor(PANEL_BG)
        top5    = top_idx[:5]
        sit_pos = np.arange(len(SITUATION_NAMES))
        bar_w   = 0.15
        palette = ["#4FC3F7", "#81C784", "#FFB74D", "#F06292", "#CE93D8"]
        for rank, nidx in enumerate(top5):
            offset = (rank - 2) * bar_w
            ax2.bar(sit_pos + offset, means[nidx],
                    width=bar_w * 0.9, color=palette[rank], alpha=0.85,
                    label=f"Neurone #{nidx} (score={spec_score[nidx]:.2f})",
                    edgecolor="#0D1117")
        ax2.set_xticks(sit_pos)
        ax2.set_xticklabels(
            [s.replace("\n", " ") for s in SITUATION_NAMES],
            rotation=35, ha="right", color=TEXT_COL, fontsize=7.5
        )
        ax2.legend(fontsize=7, facecolor="#0D1117", edgecolor="#444",
                   labelcolor="white", loc="upper right")
        apply_style(ax2, title="Profil des 5 neurones les plus spécialisés",
                    ylabel="Activation moyenne")
        ax2.grid(axis="y", color=GRID_COL, lw=0.5, alpha=0.5)

    plt.savefig(out("xai_specialization.png"), dpi=150,
                bbox_inches="tight", facecolor=BG)
    print(f"[XAI] Sauvegarde → {out('xai_specialization.png')}")
    plt.show()


# ─── Analyse 3 : t-SNE / UMAP ─────────────────────────────────────────────────
def _run_tsne(data: np.ndarray, perplexity: float = 30.0) -> np.ndarray:
    from sklearn.manifold import TSNE
    import sklearn
    from packaging import version
    iter_kwarg = ("max_iter" if version.parse(sklearn.__version__)
                  >= version.parse("1.4") else "n_iter")
    tsne = TSNE(n_components=2,
                perplexity=min(perplexity, data.shape[0] - 1),
                learning_rate="auto", init="pca",
                random_state=42, **{iter_kwarg: 1000})
    return tsne.fit_transform(data)


def _run_umap(data: np.ndarray) -> np.ndarray:
    try:
        import umap
        return umap.UMAP(n_components=2, n_neighbors=15,
                         min_dist=0.1, random_state=42).fit_transform(data)
    except ImportError:
        print("  [WARN] umap-learn non installé → fallback t-SNE")
        return _run_tsne(data)


def plot_projection(act_arrays: dict, situations: list,
                    actions: list, scores: list, method: str = "tsne"):
    situations_arr = np.array(situations)
    actions_arr    = np.array(actions)
    scores_arr     = np.array(scores, dtype=float)
    method_label   = "t-SNE" if method == "tsne" else "UMAP"

    MAX_POINTS = 3000
    T          = len(situations)
    idx        = np.sort(np.random.choice(T, min(MAX_POINTS, T), replace=False))

    fig = plt.figure(figsize=(24, 8 * len(LAYER_KEYS)), facecolor=BG)
    fig.suptitle(
        f"Projection {method_label} des activations internes – PPO\n"
        "Gauche : situation  |  Centre : action choisie  |  Droite : score",
        fontsize=13, fontweight="bold", color="white", y=1.005
    )
    gs = gridspec.GridSpec(len(LAYER_KEYS), 3, figure=fig,
                           wspace=0.30, hspace=0.55)

    SIT_COLORS = {0: "#E74C3C", 1: "#F39C12", 2: "#F1C40F", 3: "#2ECC71",
                  4: "#3498DB", 5: "#9B59B6", 6: "#1ABC9C", 7: "#95A5A6"}
    CMAP_SCORE = LinearSegmentedColormap.from_list(
        "score", ["#0D1B2A", "#1F618D", "#2ECC71", "#F39C12", "#E74C3C"])

    for row, layer_name in enumerate(LAYER_KEYS):
        acts_sub    = act_arrays[layer_name][idx]
        sits_sub    = situations_arr[idx]
        actions_sub = actions_arr[idx]
        scores_sub  = scores_arr[idx]

        print(f"  [{method_label}] {layer_name.strip()} – "
              f"{acts_sub.shape[0]} pts × {acts_sub.shape[1]} dims…")
        proj = _run_tsne(acts_sub) if method == "tsne" else _run_umap(acts_sub)
        x, y = proj[:, 0], proj[:, 1]
        ALPHA, SIZE = 0.55, 8

        # Gauche : par situation
        ax0 = fig.add_subplot(gs[row, 0])
        ax0.set_facecolor(PANEL_BG)
        for si, sname in enumerate(SITUATION_NAMES):
            mask = sits_sub == si
            if mask.sum() == 0:
                continue
            ax0.scatter(x[mask], y[mask], c=SIT_COLORS[si], s=SIZE, alpha=ALPHA,
                        label=f"{sname.replace(chr(10), ' ')} ({mask.sum()})",
                        edgecolors="none")
        ax0.legend(fontsize=6.5, facecolor="#0D1117", edgecolor="#444",
                   labelcolor="white", markerscale=2, loc="best", framealpha=0.85)
        apply_style(ax0,
                    title=f"{layer_name.strip()} – {method_label}\nColoré par situation",
                    xlabel=f"{method_label}-1", ylabel=f"{method_label}-2")

        # Centre : par action choisie
        ax1 = fig.add_subplot(gs[row, 1])
        ax1.set_facecolor(PANEL_BG)
        for ai, aname in enumerate(ACTION_NAMES):
            mask = actions_sub == ai
            if mask.sum() == 0:
                continue
            ax1.scatter(x[mask], y[mask], c=ACTION_COLORS[ai], s=SIZE, alpha=ALPHA,
                        label=f"{aname} ({mask.sum()})", edgecolors="none")
        ax1.legend(fontsize=7, facecolor="#0D1117", edgecolor="#444",
                   labelcolor="white", markerscale=2, loc="best", framealpha=0.85)
        apply_style(ax1,
                    title=f"{layer_name.strip()} – {method_label}\nColoré par action",
                    xlabel=f"{method_label}-1", ylabel=f"{method_label}-2")

        # Droite : par score
        ax2 = fig.add_subplot(gs[row, 2])
        ax2.set_facecolor(PANEL_BG)
        sc = ax2.scatter(x, y, c=scores_sub, cmap=CMAP_SCORE, s=SIZE, alpha=ALPHA,
                         edgecolors="none",
                         vmin=scores_sub.min(), vmax=max(scores_sub.max(), 1))
        cbar = plt.colorbar(sc, ax=ax2, fraction=0.04, pad=0.03)
        cbar.set_label("Score courant", color=TEXT_COL, fontsize=8)
        cbar.ax.yaxis.set_tick_params(color=TEXT_COL, labelsize=7)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COL)
        apply_style(ax2,
                    title=f"{layer_name.strip()} – {method_label}\nColoré par score",
                    xlabel=f"{method_label}-1", ylabel=f"{method_label}-2")

    fname = f"xai_{method}.png"
    plt.savefig(out(fname), dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"[XAI] Sauvegarde → {out(fname)}")
    plt.show()


# ─── Point d'entrée ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="XAI – Activations PPO Snake")
    parser.add_argument("--distribution",   action="store_true")
    parser.add_argument("--specialization", action="store_true")
    parser.add_argument("--tsne",           action="store_true")
    parser.add_argument("--umap",           action="store_true")
    parser.add_argument("--model",    type=str, default="model_best.pth",
                        help="Chemin du modèle PPO (défaut : model_best.pth)")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    run_all = not (args.distribution or args.specialization
                   or args.tsne or args.umap)

    agent     = load_agent(args.model)
    env       = SnakeEnv(render=False)
    collector = ActivationCollector(agent)

    print(f"\n[XAI] Collecte sur {args.episodes} épisode(s)…")
    states, actions, situations, scores = collect_episodes(
        agent, env, collector, args.episodes
    )
    act_arrays = collector.get_arrays()
    collector.remove()
    env.close()
    print(f"[XAI] {len(actions)} steps collectés.\n")

    if run_all or args.distribution:
        print("[XAI] ── Distribution des activations ──")
        plot_distribution(act_arrays)

    if run_all or args.specialization:
        print("[XAI] ── Neurones spécialisés ──")
        spec = compute_specialization(act_arrays, situations)
        plot_specialization(spec, situations, act_arrays)

    if run_all or args.tsne:
        print("[XAI] ── t-SNE ──")
        try:
            from sklearn.manifold import TSNE   # noqa
            plot_projection(act_arrays, situations, actions, scores, method="tsne")
        except ImportError:
            print("  [WARN] scikit-learn non installé : pip install scikit-learn")

    if run_all or args.umap:
        print("[XAI] ── UMAP ──")
        plot_projection(act_arrays, situations, actions, scores, method="umap")

    print("\n[XAI] Analyse terminée.")


if __name__ == "__main__":
    main()

# ── Tests recommandés ──────────────────────────────────────────────────────────
# Test rapide (~30s) :
#   python xai_activations_ppo.py --distribution --episodes 5
#   → Vérifie : xai_activations_ppo/xai_distribution.png
#     ✓ 4 lignes de graphes (partagée1, partagée2, actor, critic)
#     ✓ Histogrammes centrés entre -1 et 1
#     ✓ Heatmaps temporelles colorées
#
# Test spécialisation (~1min) :
#   python xai_activations_ppo.py --specialization --episodes 10
#   → Vérifie : xai_activations_ppo/xai_specialization.png
#
# Test t-SNE (lent, ~3-5min) :
#   python xai_activations_ppo.py --tsne --episodes 15
#   → Vérifie : xai_activations_ppo/xai_tsne.png
#     ✓ Clusters visibles par action/situation
#
# UMAP (plus rapide que t-SNE, nécessite : pip install umap-learn) :
#   python xai_activations_ppo.py --umap --episodes 15
