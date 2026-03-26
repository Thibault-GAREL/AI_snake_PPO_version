"""
main.py v4
==========
  python main.py                    # entraînement silencieux
  python main.py --show-every 100  # rendu tous les 100 épisodes
  python main.py --load            # reprend depuis model_best.pth
  python main.py --eval            # évaluation visuelle greedy
"""

import argparse
import os
import csv
import time
from collections import deque

import numpy as np
import torch

from PPO import PPOAgent, SnakeEnv


CFG = {
    "total_timesteps" : 8_000_000,   # v4 : budget étendu pour les 26 features
    "n_steps"         : 2048,
    "hidden_size"     : 256,
    "path_best"       : "model_best.pth",
    "path_last"       : "model_last.pth",
    "log_file"        : "training_log.csv",
    "eval_fps"        : 12,
    "train_show_fps"  : 25,
    "log_every"       : 10,
    "window_size"     : 50,
    "entropy_warn"    : 0.35,
}


def parse_args():
    p = argparse.ArgumentParser(description="PPO Snake v3")
    p.add_argument("--show-every",    type=int,   default=0,                      metavar="N")
    p.add_argument("--load",          action="store_true")
    p.add_argument("--eval",          action="store_true")
    p.add_argument("--timesteps",     type=int,   default=CFG["total_timesteps"])
    p.add_argument("--eval-episodes", type=int,   default=20)
    p.add_argument("--device",        type=str,   default=None, choices=["cpu","cuda"])
    return p.parse_args()


def _bar(mean_score: float, width: int = 20) -> str:
    fill = min(int(mean_score * 2), width)
    return "█" * fill + "░" * (width - fill)

def _ent_str(ent: float, warn: float) -> str:
    return f"{ent:.3f}{'  ⚠' if ent < warn else '   '}"


def train(args):
    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print(f"  Snake PPO v4 — Entraînement")
    print(f"  Device     : {device}")
    print(f"  Steps max  : {args.timesteps:,}")
    print(f"  OBS dim    : {SnakeEnv.OBS_DIM}  (distances + direction + longueur + urgence + food_xy + danger_binary)")
    print("=" * 65)

    agent = PPOAgent(
        obs_dim         = SnakeEnv.OBS_DIM,
        act_dim         = SnakeEnv.ACT_DIM,
        hidden          = CFG["hidden_size"],
        device          = device,
        total_timesteps = args.timesteps,
    )
    agent.N_STEPS = CFG["n_steps"]

    if args.load and os.path.exists(CFG["path_best"]):
        agent.load(CFG["path_best"])

    env = SnakeEnv(render=False)

    fields = ["episode","score","steps","mean_score_50",
              "loss_total","loss_policy","loss_value","entropy","lr","n_updates","elapsed_s"]
    log_f  = open(CFG["log_file"], "w", newline="")
    writer = csv.DictWriter(log_f, fieldnames=fields)
    writer.writeheader()

    score_window = deque(maxlen=CFG["window_size"])
    best_mean    = -float("inf")
    episode      = 0
    global_step  = 0
    last_metrics = {}
    t_start      = time.time()

    obs = env.reset()

    while global_step < args.timesteps:
        episode += 1

        show_this = args.show_every > 0 and episode % args.show_every == 0
        if show_this and not env.render_mode:
            env.close(); env = SnakeEnv(render=True); obs = env.reset()
        elif not show_this and env.render_mode:
            env.close(); env = SnakeEnv(render=False); obs = env.reset()

        ep_score = 0
        done     = False

        while not done:
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            if env.render_mode:
                env.render(fps=CFG["train_show_fps"])

            agent.buffer.push(obs, action, log_prob, reward, value, done)
            obs          = next_obs
            ep_score     = info["score"]
            global_step += 1

            if len(agent.buffer) >= agent.N_STEPS:
                last_metrics = agent.update(obs, done)

        if len(agent.buffer) > 0:
            last_metrics = agent.update(obs, done)

        obs = env.reset()

        score_window.append(ep_score)
        mean_score = np.mean(score_window)

        if len(score_window) >= 10 and mean_score > best_mean:
            best_mean = mean_score
            agent.save(CFG["path_best"])

        elapsed = time.time() - t_start
        row = {
            "episode"       : episode,
            "score"         : ep_score,
            "steps"         : global_step,
            "mean_score_50" : round(mean_score, 3),
            "loss_total"    : round(last_metrics.get("loss_total",  0), 5),
            "loss_policy"   : round(last_metrics.get("loss_policy", 0), 5),
            "loss_value"    : round(last_metrics.get("loss_value",  0), 5),
            "entropy"       : round(last_metrics.get("entropy",     0), 5),
            "lr"            : round(last_metrics.get("lr",          0), 7),
            "n_updates"     : last_metrics.get("n_updates", 0),
            "elapsed_s"     : round(elapsed, 1),
        }
        writer.writerow(row)
        log_f.flush()

        if episode % CFG["log_every"] == 0:
            steps_k = global_step / 1_000
            lr_s    = f"{last_metrics.get('lr', 0):.2e}"
            ent_s   = _ent_str(last_metrics.get("entropy", 0), CFG["entropy_warn"])
            print(
                f"  Ep {episode:>6} | "
                f"Score {ep_score:>3} | "
                f"Moy50 {mean_score:>5.2f} |{_bar(mean_score)}| "
                f"Steps {steps_k:>7.1f}k | "
                f"Ent {ent_s}| "
                f"LR {lr_s} | "
                f"{elapsed:.0f}s"
            )

    agent.save(CFG["path_last"])
    log_f.close()
    env.close()

    print("\n" + "=" * 65)
    print(f"  Entraînement terminé !")
    print(f"  Meilleur score moyen (50 ep) : {best_mean:.2f}")
    print(f"  Checkpoints : {CFG['path_best']}  {CFG['path_last']}")
    print(f"  Log CSV     : {CFG['log_file']}")
    print("=" * 65)


def evaluate(args):
    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print(f"  Snake PPO v4 — Évaluation")
    print(f"  Modèle : {CFG['path_best']}  |  Épisodes : {args.eval_episodes}")
    print("=" * 65)

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
    print(f"\n  Score moyen : {np.mean(scores):.2f}  |  Max : {max(scores)}")


if __name__ == "__main__":
    args = parse_args()
    if args.eval:
        evaluate(args)
    else:
        train(args)


# # Entraînement rapide (silencieux)
# python main.py
#
# # Avec rendu visuel tous les 100 épisodes
# python main.py --show-every 100
#
# # Reprendre depuis le meilleur checkpoint
# python main.py --load
#
# # Évaluation visuelle greedy
# python main.py --eval