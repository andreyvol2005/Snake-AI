"""Кодировщики состояния игры в вектор для нейросети.

Два варианта на выбор:

* ``vector`` — 11 классических признаков относительно головы змейки
  (текущее направление, опасно/неопасно прямо/слева/справа, где яблоко).
  Учится быстро и не требует много памяти.
* ``grid`` — три карты 18×18 (тело, голова, яблоко) плюс one-hot
  направление. Больше информации, но сходится медленнее.
"""

from __future__ import annotations

import numpy as np

from .game import GRID_SIZE, DOWN, LEFT, RIGHT, UP, SnakeGame

VECTOR_STATE_SIZE = 11
GRID_CHANNELS = 3
GRID_STATE_SIZE = GRID_CHANNELS * GRID_SIZE * GRID_SIZE + 4


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
    if kind == "grid":
        return GRID_STATE_SIZE
    raise ValueError(f"Неизвестный тип состояния: {kind}")