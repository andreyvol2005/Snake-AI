"""Кодировщики состояния игры для нейросети.

Варианты:

* ``vector`` (по умолчанию) — 11 классических признаков относительно головы:
  направление, опасность прямо/влево/вправо, где яблоко.
* ``vector24`` — 24 признака: базовые 11 плюс 8 клеток вокруг головы, длина
  змейки и расстояния до стен.
* ``grid`` — «глазами»: карта поля 18×18 со значениями
  ``0`` — пусто, ``1`` — яблоко, ``-1`` — змейка. Дополнительно подаются
  голова, sin/cos направления и два BFS-поля расстояний — всего 6 каналов
  18×18. Сеть в этом режиме свёрточная, поэтому она видит всю картину и
  лучше планирует длинные обходы собственного тела.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Iterable

import numpy as np

from .game import DELTA, DOWN, GRID_SIZE, LEFT, RIGHT, UP, SnakeGame

VECTOR_STATE_SIZE = 11
VECTOR24_STATE_SIZE = 24
GRID_CHANNELS = 6
GRID_STATE_SIZE = GRID_CHANNELS * GRID_SIZE * GRID_SIZE  # для справки

# Угол направления в радианах: вправо — 0, вверх — 90°, влево — 180°, вниз — 270°.
DIRECTION_ANGLE = {RIGHT: 0.0, UP: math.pi / 2, LEFT: math.pi, DOWN: 3 * math.pi / 2}

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


def _distance_field(n: int, sources: Iterable[tuple[int, int]],
                     blocked: set[tuple[int, int]]) -> np.ndarray:
    """Поле BFS-расстояний от sources по свободным клеткам.

    Препятствия и недостижимые клетки — ``-1``, остальные значения
    нормированы на максимум в поле, чтобы сеть видела относительный путь.
    """
    field = np.full((n, n), -1.0, dtype=np.float32)
    queue: deque[tuple[int, int]] = deque()
    for x, y in sources:
        field[y, x] = 0.0
        queue.append((x, y))

    while queue:
        x, y = queue.popleft()
        next_dist = field[y, x] + 1.0
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < n and 0 <= ny < n):
                continue
            if field[ny, nx] >= 0.0 or (nx, ny) in blocked:
                continue
            field[ny, nx] = next_dist
            queue.append((nx, ny))

    peak = float(field.max())
    if peak > 0.0:
        field = np.where(field >= 0.0, field / peak, -1.0)
    return field.astype(np.float32)


def encode_grid(game: SnakeGame) -> np.ndarray:
    """Карта поля 18×18 и подсказки о путях.

    Канал 0 — ровно то, что видит змейка: ``0`` пусто, ``1`` яблоко,
    ``-1`` змейка. Дополнительно подаются:

    * канал 1 — голова (отличаем её от тела);
    * каналы 2–3 — sin/cos текущего направления;
    * канал 4 — поле кратчайших расстояний до яблока в обход тела;
    * канал 5 — поле расстояний до хвоста (подсказка о безопасном выходе).

    Поля путей считаются тем же BFS, что использует учитель, но решение
    всё равно принимает нейросеть: это дополнительные «сенсоры», а не
    готовая команда.
    """
    view = game.view
    g = view.grid
    snake = view.snake
    head_x, head_y = snake[0]

    board = np.zeros((g, g), dtype=np.float32)
    for x, y in snake:
        board[y, x] = -1.0
    food = view.food
    if food is not None:
        board[food[1], food[0]] = 1.0

    head = np.zeros((g, g), dtype=np.float32)
    head[head_y, head_x] = 1.0

    angle = DIRECTION_ANGLE[view.direction]
    direction_sin = np.full((g, g), math.sin(angle), dtype=np.float32)
    direction_cos = np.full((g, g), math.cos(angle), dtype=np.float32)

    if food is not None:
        food_field = _distance_field(g, [food], set(snake))
    else:
        food_field = np.full((g, g), -1.0, dtype=np.float32)
    # Хвост на следующем тике освободится, поэтому его клетку не блокируем.
    tail_field = _distance_field(g, [snake[-1]], set(snake[:-1]))

    return np.stack(
        [board, head, direction_sin, direction_cos, food_field, tail_field],
        axis=0,
    )


ENCODERS = {
    "vector": encode_vector,
    "vector24": encode_vector24,
    "grid": encode_grid,
}

STATE_SHAPES = {
    "vector": (VECTOR_STATE_SIZE,),
    "vector24": (VECTOR24_STATE_SIZE,),
    "grid": (GRID_CHANNELS, GRID_SIZE, GRID_SIZE),
}


def encode(game: SnakeGame, kind: str = "vector") -> np.ndarray:
    try:
        encoder = ENCODERS[kind]
    except KeyError:
        raise ValueError(f"Неизвестный тип состояния: {kind}") from None
    return encoder(game)


def state_shape(kind: str) -> tuple[int, ...]:
    try:
        return STATE_SHAPES[kind]
    except KeyError:
        raise ValueError(f"Неизвестный тип состояния: {kind}") from None


def state_size(kind: str) -> int:
    shape = state_shape(kind)
    size = 1
    for dim in shape:
        size *= dim
    return size
