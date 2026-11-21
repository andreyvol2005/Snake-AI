"""Кодировщики состояния игры в вектор для нейросети.

Три варианта на выбор:

* ``vector`` (по умолчанию) — 11 классических признаков относительно головы:
  направление, опасность прямо/влево/вправо, где яблоко. Быстро сходится и
  показывает лучший результат.
* ``vector24`` — 24 признака: те же 11 плюс 8 клеток вокруг головы, длина
  змейки и расстояния до стен. Больше «тактического зрения», но требует
  более долгого обучения.
* ``grid`` — три карты 18×18 (тело, голова, яблоко) плюс one-hot направление.
"""

from __future__ import annotations

import numpy as np

from .game import DELTA, DOWN, GRID_SIZE, LEFT, RIGHT, UP, SnakeGame

VECTOR_STATE_SIZE = 11
VECTOR24_STATE_SIZE = 24
GRID_CHANNELS = 3
GRID_STATE_SIZE = GRID_CHANNELS * GRID_SIZE * GRID_SIZE + 4

# 8 направлений вокруг головы: N, NE, E, SE, S, SW, W, NW.
NEIGHBORS = (
    (0, -1), (1, -1), (1, 0), (1, 1),
    (0, 1), (-1, 1), (-1, 0), (-1, -1),
)


def _occupied(game: SnakeGame, x: int, y: int) -> bool:
    n = game.cfg.grid
    if not (0 <= x < n and 0 <= y < n):
        return True
    return (x, y) in game.snake


def encode_vector(game: SnakeGame) -> np.ndarray:
    """11 признаков относительно головы змейки (классический Snake DQN)."""
    view = game.view
    head_x, head_y = view.snake[0]
    food = view.food
    direction = view.direction

    left = (direction - 1) % 4
    right = (direction + 1) % 4

    features = [
        # Текущее направление движения (one-hot из 4).
        direction == UP,
        direction == RIGHT,
        direction == DOWN,
        direction == LEFT,
        # Опасность в трёх направлениях, куда можно повернуть.
        game.danger(direction),
        game.danger(left),
        game.danger(right),
        # Относительное положение яблока.
        food is not None and food[0] < head_x,   # слева
        food is not None and food[0] > head_x,   # справа
        food is not None and food[1] < head_y,   # сверху
        food is not None and food[1] > head_y,   # снизу
    ]
    return np.asarray(features, dtype=np.float32)


def encode_vector24(game: SnakeGame) -> np.ndarray:
    """24 признака: 11 базовых + окрестность головы + длина + расстояния до стен."""
    view = game.view
    head_x, head_y = view.snake[0]
    n = view.grid

    features = list(encode_vector(game))

    # 8 клеток вокруг головы: 1 — тело или стена, 0 — свободно.
    for dx, dy in NEIGHBORS:
        features.append(_occupied(game, head_x + dx, head_y + dy))

    # Длина змейки и расстояния до стен.
    features.append(len(view.snake) / (n * n))
    features.extend([
        head_y / (n - 1),             # до верхней стены
        (n - 1 - head_x) / (n - 1),   # до правой
        (n - 1 - head_y) / (n - 1),   # до нижней
        head_x / (n - 1),             # до левой
    ])

    return np.asarray(features, dtype=np.float32)


def encode_grid(game: SnakeGame) -> np.ndarray:
    """Карты 18×18: тело, голова, яблоко + one-hot направление."""
    view = game.view
    g = view.grid

    body = np.zeros((g, g), dtype=np.float32)
    for x, y in view.snake[1:]:
        body[y, x] = 1.0

    head = np.zeros((g, g), dtype=np.float32)
    hx, hy = view.snake[0]
    head[hy, hx] = 1.0

    food = np.zeros((g, g), dtype=np.float32)
    if view.food is not None:
        fx, fy = view.food
        food[fy, fx] = 1.0

    direction = np.zeros(4, dtype=np.float32)
    direction[view.direction] = 1.0

    return np.concatenate([body.ravel(), head.ravel(), food.ravel(), direction])


ENCODERS = {
    "vector": encode_vector,
    "vector24": encode_vector24,
    "grid": encode_grid,
}


def encode(game: SnakeGame, kind: str = "vector") -> np.ndarray:
    try:
        encoder = ENCODERS[kind]
    except KeyError:
        raise ValueError(f"Неизвестный тип состояния: {kind}") from None
    return encoder(game)


def state_size(kind: str) -> int:
    if kind == "vector":
        return VECTOR_STATE_SIZE
    if kind == "vector24":
        return VECTOR24_STATE_SIZE
    if kind == "grid":
        return GRID_STATE_SIZE
    raise ValueError(f"Неизвестный тип состояния: {kind}")