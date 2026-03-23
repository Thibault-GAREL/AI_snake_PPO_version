"""
PPO.py v3
=========
Corrections vs v2 :

1. LR schedule corrigé : basé sur n_updates réels (pas sur timesteps/N_STEPS)
   → Le LR ne tombe plus à 0 après 1470 épisodes
   → Remplacé LinearLR par CosineAnnealingLR (plus robuste aux updates irréguliers)

2. Reward shaping corrigé :
   - Suppression du shaping Manhattan trop "tunnel-vision"
   - Remplacé par une récompense de survie + bonus nourriture fort
   - Pénalité de mort relative à la longueur du serpent
   - Pénalité de tourner en rond (steps_since_food trop élevé)

3. State enrichi : 22 features (+ longueur normalisée + urgence nourriture)
   → Le serpent sait combien il mesure et depuis combien de temps il n'a pas mangé
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import pygame
import random
from dataclasses import dataclass
from typing import Tuple, List, Optional


# ──────────────────────────────────────────────
# Snake (version autonome)
# ──────────────────────────────────────────────

@dataclass
class Snake:
    x: int
    y: int

@dataclass
class food:
    x: int
    y: int

class Manager_snake:
    def __init__(self, width: int, height: int):
        self.list_snake = []
        self.lenght     = 0
        self.direction  = "RIGHT"
        self.moved      = True
        self.width      = width
        self.height     = height

    def add_snake(self, s: Snake):
        self.list_snake.append(s)
        self.lenght += 1

    def move(self) -> bool:
        hx, hy = self.list_snake[0].x, self.list_snake[0].y
        if   self.direction == "UP":    hy -= 50
        elif self.direction == "DOWN":  hy += 50
        elif self.direction == "RIGHT": hx += 50
        elif self.direction == "LEFT":  hx -= 50

        new_head = Snake(hx, hy)
        if not (0 <= new_head.x < self.width and 0 <= new_head.y < self.height):
            return False
        if new_head in self.list_snake:
            return False

        self.list_snake.insert(0, new_head)
        self.list_snake.pop(-1)
        self.moved = True
        return True

    def draw_snake(self, display, cell, GREEN=(0, 255, 0), BLACK=(40, 40, 60)):
        for s in self.list_snake:
            pygame.draw.rect(display, GREEN, (s.x, s.y, cell, cell))


# ──────────────────────────────────────────────
# Fonctions de distance
# ──────────────────────────────────────────────

def _dist_north(s, h):
    d = s.list_snake[0].y
    for seg in s.list_snake:
        if seg.x == s.list_snake[0].x and 0 < s.list_snake[0].y - seg.y < d:
            d = s.list_snake[0].y - seg.y
    return d

def _dist_south(s, h):
    d = h - 50 - s.list_snake[0].y
    for seg in s.list_snake:
        if seg.x == s.list_snake[0].x and 0 < seg.y - s.list_snake[0].y < d:
            d = seg.y - s.list_snake[0].y
    return d

def _dist_west(s, w):
    d = s.list_snake[0].x
    for seg in s.list_snake:
        if seg.y == s.list_snake[0].y and 0 < s.list_snake[0].x - seg.x < d:
            d = s.list_snake[0].x - seg.x
    return d

def _dist_east(s, w):
    d = w - 50 - s.list_snake[0].x
    for seg in s.list_snake:
        if seg.y == s.list_snake[0].y and 0 < seg.x - s.list_snake[0].x < d:
            d = seg.x - s.list_snake[0].x
    return d

def _dist_diag(s, w, h, dx, dy):
    hx, hy  = s.list_snake[0].x, s.list_snake[0].y
    steps_x = (w - 50 - hx) // 50 if dx > 0 else hx // 50
    steps_y = (h - 50 - hy) // 50 if dy > 0 else hy // 50
    steps   = min(steps_x, steps_y)
    d       = math.sqrt(2) * steps * 50
    for seg in s.list_snake:
        sdx, sdy = seg.x - hx, seg.y - hy
        if sdx == 0 or sdy == 0:
            continue
        if (sdx > 0) == (dx > 0) and (sdy > 0) == (dy > 0) and abs(sdx) == abs(sdy):
            dist = math.sqrt(sdx**2 + sdy**2)
            if dist < d:
                d = dist
    return d

def _food_distances(s, f):
    hx, hy = s.list_snake[0].x, s.list_snake[0].y
    fx, fy = f.x, f.y
    n = ne = e = se = sm = sw = w = nw = 0.0
    if fy < hy and fx == hx: n  = hy - fy
    if fx > hx and fy == hy: e  = fx - hx
    if fy > hy and fx == hx: sm = fy - hy
    if fx < hx and fy == hy: w  = hx - fx
    if fx != hx and fy != hy:
        dist = math.sqrt((fx-hx)**2 + (fy-hy)**2)
        try:
            p = (hx - fx) / (hy - fy)
        except ZeroDivisionError:
            p = None
        if p == -1 and hx < fx:  ne = dist
        elif p ==  1 and hx < fx: se = dist
        elif p == -1 and fx < hx: sw = dist
        elif p ==  1 and fx < hx: nw = dist
    return n, ne, e, se, sm, sw, w, nw


# ──────────────────────────────────────────────
# Environnement v3
# ──────────────────────────────────────────────

class SnakeEnv:
    """
    State (22 features) :
      [0:8]  distances aux dangers (N NE E SE S SW W NW), normalisées
      [8:16] distances nourriture (même 8 directions), normalisées
      [16:20] one-hot direction courante
      [20]   longueur serpent / max_length (0→1)
      [21]   urgence nourriture : steps_since_food / MAX_STEPS (0→1)
    """

    WIDTH     = 800
    HEIGHT    = 400
    CELL      = 50
    COLS      = WIDTH  // CELL   # 16
    ROWS      = HEIGHT // CELL   # 8
    MAX_CELLS = COLS * ROWS      # 128 (longueur maximale théorique)
    MAX_STEPS = 500
    OBS_DIM   = 22               # ← +2 vs v2
    ACT_DIM   = 4

    _MAX_DIST = math.sqrt(WIDTH**2 + HEIGHT**2)
    _DIR_IDX  = {"UP": 0, "RIGHT": 1, "DOWN": 2, "LEFT": 3}

    def __init__(self, render: bool = False):
        self.render_mode     = render
        self.display         = None
        self.clock           = None
        self._step           = 0
        self._score          = 0
        self._snake          = None
        self._food           = None
        self._steps_since_food = 0

        if render:
            pygame.init()
            self.display = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Snake PPO v3")
            self.clock   = pygame.time.Clock()
            self.font    = pygame.font.SysFont(None, 30)

    def _gen_food(self):
        occupied = {(sg.x, sg.y) for sg in self._snake.list_snake}
        free = [
            (x * self.CELL, y * self.CELL)
            for x in range(self.COLS) for y in range(self.ROWS)
            if (x * self.CELL, y * self.CELL) not in occupied
        ]
        if not free:
            return None
        rx, ry = random.choice(free)
        return food(rx, ry)

    def _get_state(self) -> np.ndarray:
        s, f = self._snake, self._food
        w, h = self.WIDTH, self.HEIGHT
        M    = self._MAX_DIST

        dn  = _dist_north(s, h)
        ds  = _dist_south(s, h)
        dw  = _dist_west(s, w)
        de  = _dist_east(s, w)
        dne = _dist_diag(s, w, h, +1, -1)
        dse = _dist_diag(s, w, h, +1, +1)
        dsw = _dist_diag(s, w, h, -1, +1)
        dnw = _dist_diag(s, w, h, -1, -1)

        fn, fne, fe, fse, fsm, fsw, fw, fnw = _food_distances(s, f)

        dir_oh = [0.0, 0.0, 0.0, 0.0]
        dir_oh[self._DIR_IDX[s.direction]] = 1.0

        length_norm = (s.lenght - 1) / (self.MAX_CELLS - 1)
        urgency     = min(self._steps_since_food / self.MAX_STEPS, 1.0)

        raw = np.array([
            dn, dne, de, dse, ds, dsw, dw, dnw,
            fn, fne, fe, fse, fsm, fsw, fw, fnw,
            *dir_oh,
            length_norm,
            urgency,
        ], dtype=np.float32)

        raw[:16] /= M
        return raw

    def reset(self) -> np.ndarray:
        self._snake = Manager_snake(self.WIDTH, self.HEIGHT)
        self._snake.add_snake(Snake(5 * self.CELL, 5 * self.CELL))
        self._food             = self._gen_food()
        self._step             = 0
        self._score            = 0
        self._steps_since_food = 0
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        self._step            += 1
        self._steps_since_food += 1

        dir_map   = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}
        opposites = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
        new_dir   = dir_map[action]
        if new_dir != opposites.get(self._snake.direction, ""):
            self._snake.direction = new_dir

        fx, fy = self._food.x, self._food.y
        alive  = self._snake.move()

        if not alive:
            # Pénalité de mort proportionnelle à la longueur (perdre un grand serpent = pire)
            death_penalty = -10.0 - self._snake.lenght * 0.5
            return self._get_state(), death_penalty, True, {"score": self._score}

        nhx, nhy = self._snake.list_snake[0].x, self._snake.list_snake[0].y

        # Nourriture mangée
        if nhx == fx and nhy == fy:
            tail = Snake(self._snake.list_snake[-1].x, self._snake.list_snake[-1].y)
            self._snake.add_snake(tail)
            self._food             = self._gen_food()
            self._score           += 1
            self._steps_since_food = 0
            if self._food is None:
                return self._get_state(), 20.0, True, {"score": self._score}
            return self._get_state(), 10.0, False, {"score": self._score}

        # Récompense de survie simple
        reward = 0.02

        # Pénalité si le serpent tourne en rond trop longtemps sans manger
        # (proportionnelle à sa longueur : plus il est long, plus c'est dangereux)
        max_allowed = self.MAX_STEPS - self._snake.lenght * 2
        if self._steps_since_food > max_allowed:
            reward -= 0.5

        done = (self._step >= self.MAX_STEPS)
        return self._get_state(), reward, done, {"score": self._score}

    def render(self, fps: int = 15):
        if not self.render_mode or self.display is None:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
        BLACK, GREEN, RED, WHITE = (40,40,60), (0,255,0), (255,0,0), (255,255,255)
        self.display.fill(BLACK)
        pygame.draw.rect(self.display, RED, (self._food.x, self._food.y, self.CELL, self.CELL))
        self._snake.draw_snake(self.display, self.CELL, GREEN, BLACK)
        txt = self.font.render(
            f"Score: {self._score}  Len: {self._snake.lenght}  Steps: {self._step}", True, WHITE
        )
        self.display.blit(txt, (10, 10))
        pygame.display.update()
        self.clock.tick(fps)

    def close(self):
        if self.render_mode:
            pygame.quit()


# ──────────────────────────────────────────────
# Réseau Actor-Critic v3
# ──────────────────────────────────────────────

class ActorCritic(nn.Module):
    """
    22 → 256 → 256 → (4 logits | 1 valeur)
    Tronc avec LayerNorm + Tanh (stable pour PPO)
    """

    def __init__(self, obs_dim: int = 22, act_dim: int = 4, hidden: int = 256):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
        )
        self.actor_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.Tanh(),
            nn.Linear(hidden // 2, act_dim),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.Tanh(),
            nn.Linear(hidden // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.actor_head[-1].weight,  gain=0.01)
        nn.init.orthogonal_(self.critic_head[-1].weight, gain=1.0)

    def forward(self, x):
        f = self.shared(x)
        return self.actor_head(f), self.critic_head(f).squeeze(-1)

    def get_action(self, obs, deterministic=False):
        logits, value = self(obs)
        dist  = Categorical(logits=logits)
        act   = logits.argmax(-1) if deterministic else dist.sample()
        return act, dist.log_prob(act), value, dist.entropy()


# ──────────────────────────────────────────────
# Buffer de rollout
# ──────────────────────────────────────────────

class RolloutBuffer:
    def __init__(self):
        self.obs:       List[np.ndarray] = []
        self.actions:   List[int]        = []
        self.log_probs: List[float]      = []
        self.rewards:   List[float]      = []
        self.values:    List[float]      = []
        self.dones:     List[bool]       = []

    def push(self, obs, action, log_prob, reward, value, done):
        self.obs.append(obs); self.actions.append(action)
        self.log_probs.append(log_prob); self.rewards.append(reward)
        self.values.append(value); self.dones.append(done)

    def clear(self): self.__init__()
    def __len__(self): return len(self.rewards)

    def compute_returns_advantages(self, last_value, gamma, gae_lambda, device):
        n   = len(self.rewards)
        adv = np.zeros(n, dtype=np.float32)
        gae = 0.0
        v   = np.array(self.values + [last_value], dtype=np.float32)
        d   = np.array(self.dones,                 dtype=np.float32)

        for t in reversed(range(n)):
            delta  = self.rewards[t] + gamma * v[t+1] * (1 - d[t]) - v[t]
            gae    = delta + gamma * gae_lambda * (1 - d[t]) * gae
            adv[t] = gae

        ret   = adv + np.array(self.values, dtype=np.float32)
        obs_t = torch.tensor(np.array(self.obs),       dtype=torch.float32, device=device)
        act_t = torch.tensor(np.array(self.actions),   dtype=torch.long,    device=device)
        lp_t  = torch.tensor(np.array(self.log_probs), dtype=torch.float32, device=device)
        adv_t = torch.tensor(adv,                      dtype=torch.float32, device=device)
        ret_t = torch.tensor(ret,                      dtype=torch.float32, device=device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        return obs_t, act_t, lp_t, adv_t, ret_t


# ──────────────────────────────────────────────
# Agent PPO v3
# ──────────────────────────────────────────────

class PPOAgent:
    """
    PPO v3 — corrections clés :

    LR schedule : CosineAnnealingLR (T_max = total_timesteps, eta_min = 1e-5)
      → Décroissance douce sur toute la durée, jamais à 0
      → Robuste au fait que chaque épisode peut déclencher une update

    Hyperparamètres :
      LR         = 3e-4
      GAMMA      = 0.99
      GAE_LAMBDA = 0.95
      CLIP_EPS   = 0.15
      ENT_COEF   = 0.05
      VF_COEF    = 0.5
      MAX_GRAD   = 0.5
      N_EPOCHS   = 8
      BATCH_SIZE = 64
      N_STEPS    = 2048
    """

    LR         = 3e-4
    GAMMA      = 0.99
    GAE_LAMBDA = 0.95
    CLIP_EPS   = 0.15
    ENT_COEF   = 0.05
    VF_COEF    = 0.5
    MAX_GRAD   = 0.5
    N_EPOCHS   = 8
    BATCH_SIZE = 64
    N_STEPS    = 2048

    def __init__(self, obs_dim: int = 22, act_dim: int = 4,
                 hidden: int = 256, device: Optional[torch.device] = None,
                 total_timesteps: int = 3_000_000):

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"[PPO v3] Device : {self.device}")

        self.net   = ActorCritic(obs_dim, act_dim, hidden).to(self.device)
        self.optim = optim.Adam(self.net.parameters(), lr=self.LR, eps=1e-5)

        # ── CosineAnnealingLR : descend en cosinus sur tout l'entraînement
        # T_max = nombre d'updates total estimé (plus conservateur)
        # eta_min = 1e-5 (jamais à 0, permet de continuer à apprendre)
        n_updates_total = max(1, total_timesteps // self.N_STEPS) * 3  # x3 : marge pour les updates résiduelles
        self.scheduler  = optim.lr_scheduler.CosineAnnealingLR(
            self.optim,
            T_max   = n_updates_total,
            eta_min = 1e-5,
        )

        self.buffer     = RolloutBuffer()
        self._n_updates = 0

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, deterministic: bool = False):
        obs_t  = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        act, lp, val, _ = self.net.get_action(obs_t, deterministic)
        return act.item(), lp.item(), val.item()

    def update(self, last_obs: np.ndarray, last_done: bool) -> dict:
        with torch.no_grad():
            lt    = torch.tensor(last_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            _, lv = self.net(lt)
            lv_val = lv.item() * (1 - float(last_done))

        obs_t, act_t, lp_t, adv_t, ret_t = \
            self.buffer.compute_returns_advantages(lv_val, self.GAMMA, self.GAE_LAMBDA, self.device)

        n   = len(self.buffer)
        idx = np.arange(n)
        agg = {"pg": 0.0, "vf": 0.0, "ent": 0.0, "tot": 0.0}
        cnt = 0

        for _ in range(self.N_EPOCHS):
            np.random.shuffle(idx)
            for start in range(0, n, self.BATCH_SIZE):
                bi = idx[start: start + self.BATCH_SIZE]
                logits, values = self.net(obs_t[bi])
                dist    = Categorical(logits=logits)
                new_lp  = dist.log_prob(act_t[bi])
                entropy = dist.entropy().mean()

                ratio  = torch.exp(new_lp - lp_t[bi])
                pg     = torch.max(
                    -adv_t[bi] * ratio,
                    -adv_t[bi] * torch.clamp(ratio, 1 - self.CLIP_EPS, 1 + self.CLIP_EPS)
                ).mean()
                vf     = nn.functional.mse_loss(values, ret_t[bi])
                loss   = pg + self.VF_COEF * vf - self.ENT_COEF * entropy

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.MAX_GRAD)
                self.optim.step()

                agg["pg"] += pg.item(); agg["vf"] += vf.item()
                agg["ent"] += entropy.item(); agg["tot"] += loss.item()
                cnt += 1

        self.scheduler.step()
        self.buffer.clear()
        self._n_updates += 1
        cnt = max(cnt, 1)

        return {
            "loss_total"  : agg["tot"] / cnt,
            "loss_policy" : agg["pg"]  / cnt,
            "loss_value"  : agg["vf"]  / cnt,
            "entropy"     : agg["ent"] / cnt,
            "n_updates"   : self._n_updates,
            "lr"          : self.scheduler.get_last_lr()[0],
        }

    def save(self, path: str):
        torch.save({
            "model_state" : self.net.state_dict(),
            "optim_state" : self.optim.state_dict(),
            "sched_state" : self.scheduler.state_dict(),
            "n_updates"   : self._n_updates,
        }, path)
        print(f"[PPO v3] Sauvegardé → {path}")

    def load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Introuvable : {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["model_state"])
        self.optim.load_state_dict(ckpt["optim_state"])
        if "sched_state" in ckpt:
            self.scheduler.load_state_dict(ckpt["sched_state"])
        self._n_updates = ckpt.get("n_updates", 0)
        print(f"[PPO v3] Chargé ← {path}  (updates : {self._n_updates})")
