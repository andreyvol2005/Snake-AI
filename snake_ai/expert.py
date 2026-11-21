"""Безопасный BFS-планировщик: учитель для обучения и «щит» при игре.

Планировщик для каждого из трёх действий считает:

* останется ли путь от новой головы до хвоста (не запираемся ли в тупике);
* длину кратчайшего пути к яблоку в обход тела;
* сколько свободных клеток достижимо после хода.

Эти числа образуют ключ безопасности. Во время обучения они подсказывают
сети осмысленные ходы (DQfD), а при игре служат защитным слоем: нейросеть
выбирает действие только среди ходов с лучшим ключом. Без такого щита
одна ошибка сети на длинной змейке почти всегда означает смерть.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

from .game import ACTIONS, DELTA, KEEP, TURN_L, TURN_R, TURN_LEFT, TURN_RIGHT


def _neighbors(x: int, y: int, n: int):
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < n and 0 <= ny < n:
            yield nx, ny


def _distance(n: int, start: tuple[int, int], target: tuple[int, int],
              blocked: set[tuple[int, int]]) -> Optional[int]:
    """Кратчайшее расстояние по свободным клеткам или None."""
    if start == target:
        return 0
    queue = deque([(start[0], start[1], 0)])
    seen = {start}
    while queue:
        x, y, dist = queue.popleft()
        for nx, ny in _neighbors(x, y, n):
            cell = (nx, ny)
            if cell in seen:
                continue
            if cell in blocked and cell != target:
                continue
            if cell == target:
                return dist + 1
            seen.add(cell)
            queue.append((nx, ny, dist + 1))
    return None


def _free_area(n: int, start: tuple[int, int], blocked: set[tuple[int, int]],
               cap: int = 200) -> int:
    """Сколько свободных клеток достижимо из start (обрезаем на cap)."""
    queue = deque([start])
    seen = {start}
    while queue and len(seen) < cap:
        x, y = queue.popleft()
        for nx, ny in _neighbors(x, y, n):
            cell = (nx, ny)
            if cell in seen or cell in blocked:
                continue
            seen.add(cell)
            queue.append(cell)
    return len(seen)


@dataclass(frozen=True)
class ActionOption:
    """Оценка одного хода: чем меньше key, тем ход безопаснее и короче."""

    action: int
    new_head: tuple[int, int]
    tail_ok: bool
    food_distance: int
    free_area: int

    @property
    def key(self) -> tuple[int, int, int]:
        return (
            0 if self.tail_ok else 1,   # сначала ходы с выходом к хвосту
            self.food_distance,          # затем короткий путь к яблоку
            -self.free_area,             # затем больше свободного места
        )


def action_options(game) -> list[ActionOption]:
    """Перебрать три действия и оценить каждое по BFS-критериям."""
    n = game.cfg.grid
    head = game.snake[0]
    body = set(game.snake[:-1])
    body.discard(head)

    options: list[ActionOption] = []
    for action in ACTIONS:
        direction = game.direction
        if action == TURN_L:
            direction = TURN_LEFT[direction]
        elif action == TURN_R:
            direction = TURN_RIGHT[direction]
        dx, dy = DELTA[direction]
        new_head = (head[0] + dx, head[1] + dy)
        if not (0 <= new_head[0] < n and 0 <= new_head[1] < n):
            continue
        if new_head in body:
            continue

        if new_head == game.food:
            future = [new_head] + list(game.snake)
        else:
            future = [new_head] + list(game.snake[:-1])
        future_tail = future[-1]
        future_body = set(future[:-1])
        tail_ok = _distance(n, new_head, future_tail, future_body) is not None

        food_distance = 10_000
        if game.food is not None:
            found = _distance(n, new_head, game.food, body)
            if found is not None:
                food_distance = found
        area = _free_area(n, new_head, future_body)
        options.append(
            ActionOption(action, new_head, tail_ok, food_distance, area)
        )
    return options


def best_key(options: list[ActionOption]) -> Optional[tuple[int, int, int]]:
    if not options:
        return None
    return min(option.key for option in options)


def allowed_actions(game) -> tuple[list[int], list[ActionOption]]:
    """Действия с лучшим ключом безопасности (обычно одно, иногда несколько)."""
    options = action_options(game)
    key = best_key(options)
    if key is None:
        return [KEEP], options
    return [option.action for option in options if option.key == key], options


def expert_action(game) -> int:
    """Подсказать действие: к яблоку по короткому пути, но без ловушки."""
    actions, _ = allowed_actions(game)
    return actions[0]
