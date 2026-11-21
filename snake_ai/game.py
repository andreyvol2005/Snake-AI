"""Игровая логика змейки без графики.

Один и тот же движок используется и человеком (через клавиатуру),
и ИИ-агентом (через выбор направления). Никакого pygame внутри —
поэтому обучение идёт в тысячи раз быстрее, чем с отрисовкой.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

GRID_SIZE = 18

# Направления: индекс = направление движения головы.
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
DELTA = {UP: (0, -1), RIGHT: (1, 0), DOWN: (0, 1), LEFT: (-1, 0)}
TURN_LEFT = {UP: LEFT, LEFT: DOWN, DOWN: RIGHT, RIGHT: UP}
TURN_RIGHT = {UP: RIGHT, RIGHT: DOWN, DOWN: LEFT, LEFT: UP}

# Действия агента: не менять направление, повернуть влево, повернуть вправо.
KEEP, TURN_L, TURN_R = 0, 1, 2
ACTIONS = (KEEP, TURN_L, TURN_R)

Point = Tuple[int, int]


@dataclass
class GameConfig:
    """Настройки игры и системы поощрений."""

    grid: int = GRID_SIZE
    eat_reward: float = 10.0
    # Смерть теперь стоит заметно дороже яблока: агенту невыгодно
    # рисковать длинной змейкой ради одного лишнего очка.
    death_penalty: float = -40.0
    # Награда за каждый прожитый шаг. Раньше за шаг был маленький штраф,
    # из-за чего агенту было всё равно, выживет он или умрёт.
    survival_bonus: float = 0.1
    # Поощрение за приближение к яблоку по кратчайшему пути В ОБХОД ТЕЛА
    # (BFS), а не по прямой. Прямая часто ведёт в собственный хвост.
    distance_shaping: float = 0.4
    win_reward: float = 200.0
    # Максимум тиков на партию. Должно хватать, чтобы змейка выросла до
    # длинного тела: на длинной змейке одному шагу соответствует много тиков.
    max_steps: int = 3000
    # Если яблоко не съедено за это количество тиков — партия считается
    # зацикленной и прерывается со штрафом.
    starve_steps: int = 220
    starve_penalty: float = -10.0
    # BFS-награда: вести змейку к яблоку по свободному коридору, а не
    # сквозь собственное тело. Отключается для старых экспериментов.
    bfs_shaping: bool = True


@dataclass
class GameView:
    """Снимок состояния игры, который получают кодировщики состояния."""

    snake: Tuple[Point, ...]  # голова — нулевой элемент
    direction: int
    food: Optional[Point]
    grid: int
    score: int


@dataclass
class StepResult:
    view: GameView
    reward: float
    done: bool
    ate: bool
    steps: int
    score: int
    steps_since_food: int


class SnakeGame:
    def __init__(self, config: Optional[GameConfig] = None, rng: Optional[random.Random] = None):
        self.cfg = config or GameConfig()
        self.rng = rng or random.Random()
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self) -> GameView:
        c = self.cfg.grid
        cy = c // 2
        # Голова впереди, тело тянется влево, стартовое направление — вправо.
        self.snake = [(c // 2 - i, cy) for i in range(3)]
        self.direction = RIGHT
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.done = False
        self._place_food()
        return self.view

    def _place_food(self) -> None:
        occupied = set(self.snake)
        free = [(x, y) for y in range(self.cfg.grid) for x in range(self.cfg.grid)
                if (x, y) not in occupied]
        self.food = self.rng.choice(free) if free else None

    @property
    def view(self) -> GameView:
        return GameView(
            snake=tuple(self.snake),
            direction=self.direction,
            food=self.food,
            grid=self.cfg.grid,
            score=self.score,
        )

    # ------------------------------------------------------------------- step
    def step(self, action: int) -> StepResult:
        """Продвинуть игру на один тик.

        action: KEEP / TURN_L / TURN_R. Разворот на 180° невозможен —
        агент физически не может выбрать «назад», как и человек.
        """
        if self.done:
            raise RuntimeError("Игра завершена — нужен reset()")

        if action == TURN_L:
            self.direction = TURN_LEFT[self.direction]
        elif action == TURN_R:
            self.direction = TURN_RIGHT[self.direction]
        elif action != KEEP:
            raise ValueError(f"Неизвестное действие: {action}")

        hx, hy = self.snake[0]
        dx, dy = DELTA[self.direction]
        head = (hx + dx, hy + dy)
        ate = head == self.food

        # Хвост уходит с клетки, если змейка не растёт — сталкиваться с ним можно.
        body_to_check = self.snake if ate else self.snake[:-1]
        outside = not (0 <= head[0] < self.cfg.grid and 0 <= head[1] < self.cfg.grid)

        if outside or head in body_to_check:
            self.done = True
            return StepResult(
                self.view, self.cfg.death_penalty, True, False,
                self.steps, self.score, self.steps_since_food,
            )

        old_distance = self._path_distance()
        self.snake.insert(0, head)
        self.steps += 1
        self.steps_since_food += 1

        if ate:
            self.score += 1
            self.steps_since_food = 0
            self._place_food()
            if self.food is None:  # поле заполнено — победа
                self.done = True
                return StepResult(
                    self.view, self.cfg.win_reward, True, True,
                    self.steps, self.score, self.steps_since_food,
                )
            reward = self.cfg.eat_reward + self.cfg.survival_bonus
        else:
            self.snake.pop()
            reward = self.cfg.survival_bonus
            if self.cfg.distance_shaping:
                new_distance = self._path_distance()
                if old_distance is not None and new_distance is not None:
                    reward += self.cfg.distance_shaping * (old_distance - new_distance)

        # Защита от зацикливания: если долго не ест — прерываем со штрафом.
        # Запас растёт с длиной змейки: длинной змейке нужно больше тиков,
        # чтобы добраться до яблока.
        starve_limit = self.cfg.starve_steps + 10 * len(self.snake)
        starved = self.steps_since_food >= starve_limit
        if starved and not self.done:
            self.done = True
            reward += self.cfg.starve_penalty  # штраф за зацикливание

        # Лимит тиков на партию: не даём одному эпизоду длиться бесконечно.
        # Это не смерть, поэтому дополнительного штрафа нет.
        if not self.done and self.steps >= self.cfg.max_steps:
            self.done = True

        return StepResult(
            self.view, reward, self.done, ate, self.steps, self.score, self.steps_since_food,
        )

    def _distance(self) -> int:
        """Манхэттенское расстояние до яблока (используется как запасной вариант)."""
        if self.food is None:
            return 0
        hx, hy = self.snake[0]
        return abs(self.food[0] - hx) + abs(self.food[1] - hy)

    def _path_distance(self) -> Optional[int]:
        """Длина кратчайшего пути до яблока в обход тела (BFS).

        Хвост не считается препятствием: на следующем тике он уйдёт с клетки.
        ``None`` — путь полностью перекрыт собственным телом.
        """
        if self.food is None:
            return 0
        if not self.cfg.bfs_shaping:
            return self._distance()

        n = self.cfg.grid
        start = self.snake[0]
        target = self.food
        blocked = set(self.snake[:-1])
        blocked.discard(start)

        queue = deque([(start[0], start[1], 0)])
        seen = {start}
        while queue:
            x, y, dist = queue.popleft()
            if (x, y) == target:
                return dist
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < n and 0 <= ny < n):
                    continue
                cell = (nx, ny)
                if cell in seen:
                    continue
                if cell in blocked and cell != target:
                    continue
                seen.add(cell)
                queue.append((nx, ny, dist + 1))
        return None

    # ------------------------------------------------------------------ utils
    def danger(self, direction: int) -> bool:
        """Упрётся ли голова в стену или тело при движении в direction."""
        hx, hy = self.snake[0]
        dx, dy = DELTA[direction]
        nx, ny = hx + dx, hy + dy
        if not (0 <= nx < self.cfg.grid and 0 <= ny < self.cfg.grid):
            return True
        body = self.snake if len(self.snake) > 1 else []
        return (nx, ny) in body