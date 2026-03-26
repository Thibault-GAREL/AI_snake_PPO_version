# Snake AI — Unified Input Features (28 features)

Standard feature set shared across all 4 Snake AI implementations :
- NEAT ([AI_snake_genetic_version](https://github.com/Thibault-GAREL/AI_snake_genetic_version))
- DQL ([AI_snake_DQL](https://github.com/Thibault-GAREL/AI_snake_DQL))
- PPO ([snake_PPO_V2](https://github.com/Thibault-GAREL/snake_PPO_V2))
- Decision Tree ([AI_snake_decision_tree_version](https://github.com/Thibault-GAREL/AI_snake_decision_tree_version))

---

## Why uniformize ?

1. **Fair comparison** — same inputs → performance differences come only from the algorithm
2. **XAI comparability** — SHAP values, permutation importance, correlations are directly comparable across models
3. **No downside** — unused features are naturally ignored (trees don't split on them, neural nets assign low weights, NEAT prunes connections)
4. **Maintenance** — one `_get_state()` to maintain, copied across all 4 projects

---

## Feature vector (28 features)

### Group 1 — Danger distances (8 features) → path planning

Distance to the nearest obstacle (wall or body segment) in 8 directions.
Normalized by `max_dist = sqrt(WIDTH² + HEIGHT²)` → range [0, 1].

| # | Feature | Description |
|---|---------|-------------|
| 0 | `distance_danger_N` | Distance to nearest obstacle North |
| 1 | `distance_danger_NE` | Distance to nearest obstacle North-East |
| 2 | `distance_danger_E` | Distance to nearest obstacle East |
| 3 | `distance_danger_SE` | Distance to nearest obstacle South-East |
| 4 | `distance_danger_S` | Distance to nearest obstacle South |
| 5 | `distance_danger_SW` | Distance to nearest obstacle South-West |
| 6 | `distance_danger_W` | Distance to nearest obstacle West |
| 7 | `distance_danger_NW` | Distance to nearest obstacle North-West |

### Group 2 — Food distances, sparse (8 features) → navigation when aligned

Distance to food in 8 directions. **Sparse** : non-zero only when food is exactly aligned (same row, column, or exact diagonal). Normalized by `max_dist`.

| # | Feature | Description |
|---|---------|-------------|
| 8 | `distance_food_N` | Distance to food if aligned North |
| 9 | `distance_food_NE` | Distance to food if aligned North-East |
| 10 | `distance_food_E` | Distance to food if aligned East |
| 11 | `distance_food_SE` | Distance to food if aligned South-East |
| 12 | `distance_food_S` | Distance to food if aligned South |
| 13 | `distance_food_SW` | Distance to food if aligned South-West |
| 14 | `distance_food_W` | Distance to food if aligned West |
| 15 | `distance_food_NW` | Distance to food if aligned North-West |

### Group 3 — Food direction, continuous (2 features) → navigation in all situations

Relative position of food from the snake's head. **Always non-zero** — solves the blind spot of sparse features [8:15] which are zero ~80% of the time.

| # | Feature | Description | Range |
|---|---------|-------------|-------|
| 16 | `food_delta_x` | (food.x − head.x) / WIDTH | [−1, 1] |
| 17 | `food_delta_y` | (food.y − head.y) / HEIGHT | [−1, 1] |

### Group 4 — Immediate danger, binary (4 features) → survival at next step

Binary signal : is there a wall or body segment exactly 1 cell away in each cardinal direction ? **Absolute** (N/E/S/W), not relative to the snake's current direction.

| # | Feature | Description | Values |
|---|---------|-------------|--------|
| 18 | `danger_N` | Obstacle 1 cell North | 0.0 or 1.0 |
| 19 | `danger_E` | Obstacle 1 cell East | 0.0 or 1.0 |
| 20 | `danger_S` | Obstacle 1 cell South | 0.0 or 1.0 |
| 21 | `danger_W` | Obstacle 1 cell West | 0.0 or 1.0 |

**Why absolute instead of relative (front/left/right) ?**
- The direction one-hot [22:25] already encodes the current direction — the model can infer "front = danger_N when direction = UP"
- 4 absolute features cover all directions vs 2-3 relative that miss some
- Absolute features maintain stable meaning regardless of snake orientation
- The Decision Tree uses absolute and achieves the best score (22.77 mean / 43 max)

### Group 5 — Current direction, one-hot (4 features) → orientation context

| # | Feature | Description | Values |
|---|---------|-------------|--------|
| 22 | `dir_UP` | Current direction is UP | 0.0 or 1.0 |
| 23 | `dir_RIGHT` | Current direction is RIGHT | 0.0 or 1.0 |
| 24 | `dir_DOWN` | Current direction is DOWN | 0.0 or 1.0 |
| 25 | `dir_LEFT` | Current direction is LEFT | 0.0 or 1.0 |

### Group 6 — Temporal context (2 features) → game state awareness

| # | Feature | Description | Range |
|---|---------|-------------|-------|
| 26 | `length_norm` | (snake_length − 1) / (max_cells − 1) | [0, 1] |
| 27 | `urgency` | steps_since_food / MAX_STEPS | [0, 1] |

**Why include these ?**
- `length_norm` : a long snake must navigate differently (tighter spaces, more risk)
- `urgency` : signal for RL algorithms (stagnation penalty anticipation), and general "find food now" pressure
- Not present in the original DT, but harmless — trees ignore them if not useful (no split), neural nets assign low weights, NEAT prunes connections
- Cost : 0 for tree models, negligible for neural nets

---

## Output — 4 actions

| # | Action |
|---|--------|
| 0 | `UP` |
| 1 | `RIGHT` |
| 2 | `DOWN` |
| 3 | `LEFT` |

---

## What was NOT included (and why)

| Feature | Reason for exclusion |
|---------|----------------------|
| Flood fill (accessible area) | Too expensive to compute, complexifies the state |
| Absolute head position (x, y) | Game is translation-invariant — position doesn't matter, only relative distances |
| Tail direction | Weak signal, rarely useful for decision-making |
| Manhattan distance to food | Redundant with `food_delta_x` + `food_delta_y` |
| Diagonal immediate danger | Snake moves in 4 cardinal directions only — diagonal danger is a secondary concern |

---

## Normalization notes

All features are already in a well-defined range :
- Groups 1-2 : normalized by `max_dist`, range [0, 1]
- Group 3 : range [−1, 1]
- Groups 4-5 : binary, {0.0, 1.0}
- Group 6 : range [0, 1]

No additional preprocessing needed. This normalization works for all model types :
- **Neural networks** (PPO, DQL) : all features are in [−1, 1], no scale issues
- **Tree models** (DT/XGBoost) : normalization is harmless (trees are scale-invariant)
- **NEAT** : normalized inputs help initial random weights produce meaningful outputs

---

## Migration from current implementations

| Project | Current features | Changes needed |
|---------|-----------------|----------------|
| NEAT | 16 (8 danger + 8 food sparse) | +12 : add groups 3, 4, 5, 6 |
| DQL | 16 (8 danger + 8 food sparse) | +12 : add groups 3, 4, 5, 6 |
| PPO | 26 (v4 with relative danger) | +2 : replace `danger_front/left` → `danger_N/E/S/W`, reorder |
| DT | 26 (groups 1-5, no group 6) | +2 : add `length_norm` + `urgency` |

---

_Feature set designed 2026-03-26_
_Applicable to : NEAT, DQL, PPO, Decision Tree_
