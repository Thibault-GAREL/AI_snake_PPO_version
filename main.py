"""
main.py v4 — Vectorized PPO Training
=====================================
  python main.py                    # entraînement vectorisé (8 envs)
  python main.py --load             # reprend depuis model_best.pth
  python main.py --eval             # évaluation visuelle greedy
  python main.py --eval-episodes 50 # évaluation 50 épisodes
"""

import argparse
import os
import csv
import time
from collections import deque

import numpy as np
import torch
import random

from PPO import PPOAgent, SnakeEnv, VecSnakeEnv


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

CFG = {
    "total_timesteps" : 10_000_000,
    "hidden_size"     : 256,
    "seed"            : 42,
    "path_best"       : "model_best.pth",
    "path_last"       : "model_last.pth",
    "log_file"        : "training_log.csv",
    "eval_fps"        : 12,
    "log_interval"    : 5,      # log every N updates
    "window_size"     : 100,    # rolling window for score tracking
    "save_interval"   : 20,     # save checkpoint every N updates
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def parse_args():
    p = argparse.ArgumentParser(description="PPO Snake v4 — Vectorized")
    p.add_argument("--load",          action="store_true")
    p.add_argument("--eval",          action="store_true")
    p.add_argument("--timesteps",     type=int,   default=CFG["total_timesteps"])
    p.add_argument("--eval-episodes", type=int,   default=20)
    p.add_argument("--device",        type=str,   default=None, choices=["cpu", "cuda"])
    p.add_argument("--seed",          type=int,   default=CFG["seed"])
    return p.parse_args()


def _bar(mean_score: float, width: int = 25) -> str:
    fill = min(int(mean_score), width)
    return "█" * fill + "░" * (width - fill)


# ──────────────────────────────────────────────
# Entraînement vectorisé
# ──────────────────────────────────────────────

def train(args):
    set_seed(args.seed)

    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = PPOAgent(
        obs_dim         = SnakeEnv.OBS_DIM,
        act_dim         = SnakeEnv.ACT_DIM,
        hidden          = CFG["hidden_size"],
        device          = device,
        total_timesteps = args.timesteps,
    )

    if args.load and os.path.exists(CFG["path_best"]):
        agent.load(CFG["path_best"])

    vec_env = VecSnakeEnv(n_envs=agent.N_ENVS)

    n_steps   = agent.N_STEPS
    n_envs    = agent.N_ENVS
    steps_per = n_steps * n_envs  # timesteps per collection

    print("=" * 70)
    print(f"  Snake PPO v4 — Vectorized Training")
    print(f"  Device       : {device}")
    print(f"  Timesteps    : {args.timesteps:,}")
    print(f"  N_ENVS       : {n_envs}")
    print(f"  N_STEPS      : {n_steps} (per env, {steps_per:,} total per collect)")
    print(f"  BATCH_SIZE   : {agent.BATCH_SIZE}")
    print(f"  N_EPOCHS     : {agent.N_EPOCHS}")
    print(f"  CLIP_EPS     : {agent.CLIP_EPS}")
    print(f"  ENT_COEF     : {agent.ENT_COEF}")
    print(f"  LR           : {agent.LR}")
    print(f"  Seed         : {args.seed}")
    print(f"  OBS dim      : {SnakeEnv.OBS_DIM} (unified 28-feature state)")
    print("=" * 70)

    # CSV logger
    fields = ["update", "timesteps", "episodes_done", "mean_score", "max_score",
              "loss_total", "loss_policy", "loss_value", "entropy", "clip_frac",
              "lr", "fps", "elapsed_s"]
    log_f  = open(CFG["log_file"], "w", newline="")
    writer = csv.DictWriter(log_f, fieldnames=fields)
    writer.writeheader()

    # Tracking
    score_window  = deque(maxlen=CFG["window_size"])
    best_mean     = -float("inf")
    global_step   = 0
    episodes_done = 0
    t_start       = time.time()

    obs   = vec_env.reset()                          # (n_envs, 28)
    dones = np.zeros(n_envs, dtype=np.float32)

    while global_step < args.timesteps:
        # ── Collect n_steps from all envs ──
        agent.buffer.reset()
        for step in range(n_steps):
            actions, log_probs, values = agent.select_action_batch(obs)
            next_obs, rewards, new_dones, infos = vec_env.step(actions)

            agent.buffer.push(obs, actions, log_probs, rewards, values, dones)

            obs   = next_obs
            dones = new_dones
            global_step += n_envs

            # Track completed episodes
            for info in infos:
                if "terminal_score" in info:
                    score_window.append(info["terminal_score"])
                    episodes_done += 1

        # ── PPO update ──
        metrics = agent.update(obs, dones)

        # ── Logging ──
        mean_score = np.mean(score_window) if score_window else 0.0
        max_score  = max(score_window) if score_window else 0

        # Save best model
        if len(score_window) >= 20 and mean_score > best_mean:
            best_mean = mean_score
            agent.save(CFG["path_best"])

        # Periodic checkpoint
        if metrics["n_updates"] % CFG["save_interval"] == 0:
            agent.save(CFG["path_last"])

        if metrics["n_updates"] % CFG["log_interval"] == 0:
            elapsed = time.time() - t_start
            fps     = global_step / max(elapsed, 1)
            steps_k = global_step / 1_000

            row = {
                "update"        : metrics["n_updates"],
                "timesteps"     : global_step,
                "episodes_done" : episodes_done,
                "mean_score"    : round(mean_score, 2),
                "max_score"     : max_score,
                "loss_total"    : round(metrics["loss_total"], 5),
                "loss_policy"   : round(metrics["loss_policy"], 5),
                "loss_value"    : round(metrics["loss_value"], 5),
                "entropy"       : round(metrics["entropy"], 5),
                "clip_frac"     : round(metrics["clip_frac"], 4),
                "lr"            : round(metrics["lr"], 7),
                "fps"           : int(fps),
                "elapsed_s"     : round(elapsed, 1),
            }
            writer.writerow(row)
            log_f.flush()

            print(
                f"  Upd {metrics['n_updates']:>4} | "
                f"{steps_k:>7.1f}k steps | "
                f"Ep {episodes_done:>5} | "
                f"Mean {mean_score:>5.1f} | "
                f"Max {max_score:>3} |{_bar(mean_score)}| "
                f"Ent {metrics['entropy']:.3f} | "
                f"Clip {metrics['clip_frac']:.2f} | "
                f"LR {metrics['lr']:.2e} | "
                f"{fps:,.0f} fps | "
                f"{elapsed:.0f}s"
            )

    # Final save
    agent.save(CFG["path_last"])
    log_f.close()
    vec_env.close()

    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"  Entraînement terminé !")
    print(f"  Episodes     : {episodes_done:,}")
    print(f"  Best mean    : {best_mean:.2f}")
    print(f"  Max score    : {max(score_window) if score_window else 0}")
    print(f"  Durée        : {elapsed/60:.1f} min")
    print(f"  Checkpoints  : {CFG['path_best']}  {CFG['path_last']}")
    print(f"  Log CSV      : {CFG['log_file']}")
    print("=" * 70)


# ──────────────────────────────────────────────
# Évaluation
# ──────────────────────────────────────────────

def evaluate(args):
    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print(f"  Snake PPO v4 — Évaluation")
    print(f"  Modèle : {CFG['path_best']}  |  Épisodes : {args.eval_episodes}")
    print("=" * 70)

    if not os.path.exists(CFG["path_best"]):
        print(f"[!] Modèle introuvable. Lance d'abord : python main.py")
        return

    agent = PPOAgent(
        obs_dim = SnakeEnv.OBS_DIM, act_dim = SnakeEnv.ACT_DIM,
        hidden  = CFG["hidden_size"], device = device,
    )
    agent.load(CFG["path_best"])
    agent.net.eval()

    env    = SnakeEnv(render=True)
    scores = []

    for ep in range(1, args.eval_episodes + 1):
        obs  = env.reset()
        done = False
        while not done:
            action, _, _ = agent.select_action(obs, deterministic=True)
            obs, _, done, info = env.step(action)
            env.render(fps=CFG["eval_fps"])
        scores.append(info["score"])
        print(f"  Épisode {ep:>3} : score = {info['score']}")

    env.close()
    print(f"\n  Score moyen : {np.mean(scores):.2f}  |  Max : {max(scores)}  |  Min : {min(scores)}")


if __name__ == "__main__":
    args = parse_args()
    if args.eval:
        evaluate(args)
    else:
        train(args)


# # Entraînement vectorisé (8 envs parallèles)
# python main.py
#
# # Reprendre depuis le meilleur checkpoint
# python main.py --load
#
# # Évaluation visuelle greedy
# python main.py --eval
#
# # Évaluation 50 épisodes
# python main.py --eval --eval-episodes 50
