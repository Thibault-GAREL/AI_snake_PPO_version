"""
PPO.py v4 — Vectorized
=======================
Unified 28-feature state (see input.md for full specification) :

  [0:8]   danger distances (N NE E SE S SW W NW)  — normalized
  [8:16]  food distances sparse (8 dirs)           — normalized
  [16:17] food_delta_x, food_delta_y               — continuous, always non-zero
  [18:21] danger_N, danger_E, danger_S, danger_W   — binary immediate (absolute)
  [22:25] direction one-hot (UP RIGHT DOWN LEFT)
  [26]    length_norm
  [27]    urgency

Vectorized training with N_ENVS parallel environments.
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

def _danger_binary(s, w, h):
    """Retourne 4 signaux binaires : danger immédiat à 1 case en N, E, S, W (absolu)."""
    hx, hy = s.list_snake[0].x, s.list_snake[0].y
    cell = 50
    results = []
    for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:  # N, E, S, W
        nx, ny = hx + dx * cell, hy + dy * cell
        if not (0 <= nx < w and 0 <= ny < h):
            results.append(1.0)
        elif Snake(nx, ny) in s.list_snake:
            results.append(1.0)
        else:
            results.append(0.0)
    return results  # [danger_N, danger_E, danger_S, danger_W]


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
# Environnement v4 (single)
# ──────────────────────────────────────────────

class SnakeEnv:
    """
    Unified state (28 features — see input.md) :
      [0:8]   distances danger (N NE E SE S SW W NW), normalisées
      [8:16]  distances food sparse (8 dirs), normalisées
      [16]    food_delta_x : (food.x - head.x) / WIDTH   ∈ [-1, 1]
      [17]    food_delta_y : (food.y - head.y) / HEIGHT   ∈ [-1, 1]
      [18:22] danger binaire immédiat N, E, S, W (absolu) ∈ {0, 1}
      [22:26] direction one-hot (UP RIGHT DOWN LEFT)
      [26]    length_norm ∈ [0, 1]
      [27]    urgency     ∈ [0, 1]
    """

    WIDTH     = 800
    HEIGHT    = 400
    CELL      = 50
    COLS      = WIDTH  // CELL   # 16
    ROWS      = HEIGHT // CELL   # 8
    MAX_CELLS = COLS * ROWS      # 128 (longueur maximale théorique)
    MAX_STEPS = 1000
    OBS_DIM   = 28               # unified 28-feature state
    ACT_DIM   = 4

    _MAX_DIST = math.sqrt(WIDTH**2 + HEIGHT**2)
    _DIR_IDX  = {"UP": 0, "RIGHT": 1, "DOWN": 2, "LEFT": 3}  # one-hot index

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
            pygame.display.set_caption("Snake PPO v4")
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

        # [16-17] Direction continue vers la nourriture
        food_dx = (f.x - s.list_snake[0].x) / w
        food_dy = (f.y - s.list_snake[0].y) / h

        # [18-21] Danger binaire immédiat N, E, S, W (absolu)
        danger_nesw = _danger_binary(s, w, h)

        # [22-25] Direction one-hot
        dir_oh = [0.0, 0.0, 0.0, 0.0]
        dir_oh[self._DIR_IDX[s.direction]] = 1.0

        # [26-27] Contexte temporel
        length_norm = (s.lenght - 1) / (self.MAX_CELLS - 1)
        urgency     = min(self._steps_since_food / self.MAX_STEPS, 1.0)

        raw = np.array([
            dn, dne, de, dse, ds, dsw, dw, dnw,   # [0:8]   danger distances
            fn, fne, fe, fse, fsm, fsw, fw, fnw,   # [8:16]  food distances sparse
            food_dx,                                # [16]    food_delta_x (continu)
            food_dy,                                # [17]    food_delta_y (continu)
            *danger_nesw,                           # [18:22] danger_N, E, S, W (binaire)
            *dir_oh,                                # [22:26] direction one-hot
            length_norm,                            # [26]    longueur normalisée
            urgency,                                # [27]    urgence nourriture
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

        # Sauvegarde de la distance Manhattan avant le déplacement (pour le potential shaping)
        old_hx, old_hy = self._snake.list_snake[0].x, self._snake.list_snake[0].y
        old_manhattan   = abs(old_hx - fx) + abs(old_hy - fy)

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

        # Récompense de survie + potential-based shaping (proximité nourriture)
        new_manhattan = abs(nhx - fx) + abs(nhy - fy)
        shaping = 0.1 * (old_manhattan - new_manhattan) / self.CELL
        reward  = 0.02 + shaping

        # Pénalité si le serpent tourne en rond trop longtemps sans manger
        # Pression croissante avec la longueur : plus le serpent est long, moins il a de pas avant malus
        max_allowed = max(100, 300 - self._snake.lenght * 5)
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
# Vectorized Environment (N envs en parallèle)
# ──────────────────────────────────────────────

class VecSnakeEnv:
    """Wraps N SnakeEnv instances with auto-reset on done."""

    def __init__(self, n_envs: int = 8):
        self.n_envs = n_envs
        self.envs   = [SnakeEnv(render=False) for _ in range(n_envs)]

    def reset(self) -> np.ndarray:
        """Returns (n_envs, OBS_DIM) array."""
        obs = np.stack([env.reset() for env in self.envs])
        return obs

    def step(self, actions: np.ndarray):
        """
        actions: (n_envs,) int array
        Returns: obs (n_envs, OBS_DIM), rewards (n_envs,), dones (n_envs,), infos list[dict]

        Auto-resets done envs. Returned obs is the NEW episode obs for done envs.
        infos[i]["terminal_score"] is set when env i was done (before reset).
        """
        obs_list     = []
        rewards      = np.zeros(self.n_envs, dtype=np.float32)
        dones        = np.zeros(self.n_envs, dtype=np.float32)
        infos        = [{} for _ in range(self.n_envs)]

        for i, (env, act) in enumerate(zip(self.envs, actions)):
            ob, rew, done, info = env.step(int(act))
            rewards[i] = rew
            dones[i]   = float(done)
            infos[i]   = info

            if done:
                infos[i]["terminal_score"] = info["score"]
                ob = env.reset()  # auto-reset

            obs_list.append(ob)

        obs = np.stack(obs_list)
        return obs, rewards, dones, infos

    def close(self):
        for env in self.envs:
            env.close()


# ──────────────────────────────────────────────
# Vectorized Rollout Buffer
# ──────────────────────────────────────────────

class VecRolloutBuffer:
    """Fixed-size buffer for vectorized PPO collection."""

    def __init__(self, n_steps: int, n_envs: int, obs_dim: int):
        self.n_steps = n_steps
        self.n_envs  = n_envs
        self.obs_dim = obs_dim

        self.obs       = np.zeros((n_steps, n_envs, obs_dim), dtype=np.float32)
        self.actions   = np.zeros((n_steps, n_envs),          dtype=np.int64)
        self.log_probs = np.zeros((n_steps, n_envs),          dtype=np.float32)
        self.rewards   = np.zeros((n_steps, n_envs),          dtype=np.float32)
        self.values    = np.zeros((n_steps, n_envs),          dtype=np.float32)
        self.dones     = np.zeros((n_steps, n_envs),          dtype=np.float32)
        self.ptr       = 0

    def push(self, obs, actions, log_probs, rewards, values, dones):
        """Store one timestep for all envs."""
        self.obs[self.ptr]       = obs
        self.actions[self.ptr]   = actions
        self.log_probs[self.ptr] = log_probs
        self.rewards[self.ptr]   = rewards
        self.values[self.ptr]    = values
        self.dones[self.ptr]     = dones
        self.ptr += 1

    def is_full(self) -> bool:
        return self.ptr >= self.n_steps

    def reset(self):
        self.ptr = 0

    def compute_returns_advantages(self, last_values: np.ndarray, last_dones: np.ndarray,
                                    gamma: float, gae_lambda: float, device: torch.device):
        """
        Per-env GAE computation, then flatten to (n_steps * n_envs, ...).
        last_values: (n_envs,), last_dones: (n_envs,)
        """
        advantages = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        gae = np.zeros(self.n_envs, dtype=np.float32)

        # Bootstrap from last step
        next_values = last_values
        next_dones  = last_dones

        for t in reversed(range(self.n_steps)):
            delta = self.rewards[t] + gamma * next_values * (1 - self.dones[t]) - self.values[t]
            gae   = delta + gamma * gae_lambda * (1 - self.dones[t]) * gae
            advantages[t] = gae
            next_values = self.values[t]
            # Note: dones[t] already handled in delta, no need to reset gae on done
            # because the done mask in delta + gae formula handles it correctly

        returns = advantages + self.values

        # Flatten (n_steps, n_envs) → (n_steps * n_envs)
        n = self.n_steps * self.n_envs
        obs_flat  = self.obs.reshape(n, self.obs_dim)
        act_flat  = self.actions.reshape(n)
        lp_flat   = self.log_probs.reshape(n)
        val_flat  = self.values.reshape(n)
        adv_flat  = advantages.reshape(n)
        ret_flat  = returns.reshape(n)

        # Normalize advantages
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        obs_t = torch.tensor(obs_flat, dtype=torch.float32, device=device)
        act_t = torch.tensor(act_flat, dtype=torch.long,    device=device)
        lp_t  = torch.tensor(lp_flat,  dtype=torch.float32, device=device)
        val_t = torch.tensor(val_flat, dtype=torch.float32, device=device)
        adv_t = torch.tensor(adv_flat, dtype=torch.float32, device=device)
        ret_t = torch.tensor(ret_flat, dtype=torch.float32, device=device)

        return obs_t, act_t, lp_t, val_t, adv_t, ret_t


# ──────────────────────────────────────────────
# Réseau Actor-Critic v4
# ──────────────────────────────────────────────

class ActorCritic(nn.Module):
    """
    28 → 256 → 256 → (4 logits | 1 valeur)
    Tronc avec LayerNorm + Tanh (stable pour PPO)
    """

    def __init__(self, obs_dim: int = 28, act_dim: int = 4, hidden: int = 256):
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
# Agent PPO v4 — Optimized for vectorized training
# ──────────────────────────────────────────────

class PPOAgent:
    """
    PPO v5 — vectorized, optimized for high score (target: mean ~23)

    Hyperparamètres (basé sur SB3 best practices + expériences v3/v4) :
      LR         = 3e-4        (standard PPO)
      GAMMA      = 0.99        (long horizon)
      GAE_LAMBDA = 0.95        (standard)
      CLIP_EPS   = 0.15        (plus conservateur → mises à jour plus stables)
      ENT_COEF   = 0.05        (exploration accrue → critique pour serpent long)
      VF_COEF    = 0.5         (standard)
      MAX_GRAD   = 0.5         (standard)
      N_EPOCHS   = 10          (passes par mise à jour)
      BATCH_SIZE = 256         (grands mini-batches → gradients plus stables)
      N_STEPS    = 1024        (per env, 1024 * 8 = 8192 total → meilleure estimation GAE)
      N_ENVS     = 8           (vectorized)
    """

    LR         = 3e-4
    GAMMA      = 0.99
    GAE_LAMBDA = 0.95
    CLIP_EPS   = 0.15
    ENT_COEF   = 0.05
    VF_COEF    = 0.5
    MAX_GRAD   = 0.5
    N_EPOCHS   = 10
    BATCH_SIZE = 256
    N_STEPS    = 1024   # per env
    N_ENVS     = 8

    def __init__(self, obs_dim: int = 28, act_dim: int = 4,
                 hidden: int = 256, device: Optional[torch.device] = None,
                 total_timesteps: int = 10_000_000):

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"[PPO v4] Device : {self.device}")

        self.net   = ActorCritic(obs_dim, act_dim, hidden).to(self.device)
        self.optim = optim.Adam(self.net.parameters(), lr=self.LR, eps=1e-5)

        # CosineAnnealingLR over total training
        steps_per_collect = self.N_STEPS * self.N_ENVS
        n_updates_total   = max(1, total_timesteps // steps_per_collect)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optim,
            T_max   = n_updates_total,
            eta_min = 1e-5,
        )

        self.buffer     = VecRolloutBuffer(self.N_STEPS, self.N_ENVS, obs_dim)
        self._n_updates = 0

    @torch.no_grad()
    def select_action_batch(self, obs_batch: np.ndarray, deterministic: bool = False):
        """
        obs_batch: (n_envs, obs_dim) numpy array
        Returns: actions (n_envs,), log_probs (n_envs,), values (n_envs,)
        """
        obs_t = torch.tensor(obs_batch, dtype=torch.float32, device=self.device)
        act, lp, val, _ = self.net.get_action(obs_t, deterministic)
        return act.cpu().numpy(), lp.cpu().numpy(), val.cpu().numpy()

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, deterministic: bool = False):
        """Single obs for evaluation."""
        obs_t  = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        act, lp, val, _ = self.net.get_action(obs_t, deterministic)
        return act.item(), lp.item(), val.item()

    @torch.no_grad()
    def get_values_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        """Get value estimates for a batch of observations."""
        obs_t = torch.tensor(obs_batch, dtype=torch.float32, device=self.device)
        _, values = self.net(obs_t)
        return values.cpu().numpy()

    def update(self, last_obs: np.ndarray, last_dones: np.ndarray) -> dict:
        """
        PPO update with value function clipping.
        last_obs: (n_envs, obs_dim), last_dones: (n_envs,)
        """
        last_values = self.get_values_batch(last_obs) * (1 - last_dones)

        obs_t, act_t, old_lp_t, old_val_t, adv_t, ret_t = \
            self.buffer.compute_returns_advantages(
                last_values, last_dones, self.GAMMA, self.GAE_LAMBDA, self.device
            )

        n   = obs_t.shape[0]
        idx = np.arange(n)
        agg = {"pg": 0.0, "vf": 0.0, "ent": 0.0, "tot": 0.0, "clip_frac": 0.0}
        cnt = 0

        for _ in range(self.N_EPOCHS):
            np.random.shuffle(idx)
            for start in range(0, n, self.BATCH_SIZE):
                bi = idx[start: start + self.BATCH_SIZE]
                logits, values = self.net(obs_t[bi])
                dist    = Categorical(logits=logits)
                new_lp  = dist.log_prob(act_t[bi])
                entropy = dist.entropy().mean()

                # Policy loss (clipped)
                ratio    = torch.exp(new_lp - old_lp_t[bi])
                pg_loss1 = -adv_t[bi] * ratio
                pg_loss2 = -adv_t[bi] * torch.clamp(ratio, 1 - self.CLIP_EPS, 1 + self.CLIP_EPS)
                pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (clipped — SB3 style)
                values_clipped = old_val_t[bi] + torch.clamp(
                    values - old_val_t[bi], -self.CLIP_EPS, self.CLIP_EPS
                )
                vf_loss1 = (values - ret_t[bi]) ** 2
                vf_loss2 = (values_clipped - ret_t[bi]) ** 2
                vf_loss  = 0.5 * torch.max(vf_loss1, vf_loss2).mean()

                loss = pg_loss + self.VF_COEF * vf_loss - self.ENT_COEF * entropy

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.MAX_GRAD)
                self.optim.step()

                # Tracking
                with torch.no_grad():
                    clip_frac = ((ratio - 1.0).abs() > self.CLIP_EPS).float().mean().item()
                agg["pg"]  += pg_loss.item()
                agg["vf"]  += vf_loss.item()
                agg["ent"] += entropy.item()
                agg["tot"] += loss.item()
                agg["clip_frac"] += clip_frac
                cnt += 1

        self.scheduler.step()
        self.buffer.reset()
        self._n_updates += 1
        cnt = max(cnt, 1)

        return {
            "loss_total"  : agg["tot"] / cnt,
            "loss_policy" : agg["pg"]  / cnt,
            "loss_value"  : agg["vf"]  / cnt,
            "entropy"     : agg["ent"] / cnt,
            "clip_frac"   : agg["clip_frac"] / cnt,
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
        print(f"[PPO v4] Sauvegardé → {path}")

    def load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Introuvable : {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["model_state"])
        self.optim.load_state_dict(ckpt["optim_state"])
        if "sched_state" in ckpt:
            self.scheduler.load_state_dict(ckpt["sched_state"])
        self._n_updates = ckpt.get("n_updates", 0)
        print(f"[PPO v4] Chargé ← {path}  (updates : {self._n_updates})")
