"""Выбор хода: нейросеть плюс защитный BFS-слой.

Чистая сеть при игре ошибается примерно на каждом десятом ходу. Для
короткой змейки это почти незаметно, но на длинной одна ошибка обычно
заканчивает партию. Поэтому при игре мы считаем допустимыми только ходы
с лучшим ключом безопасности (путь к хвосту, расстояние до яблока и
свободная площадь), а нейросеть выбирает лучший по Q среди них.

Это не подменяет сеть: она по-прежнему оценивает каждое действие и
принимает решение, но не может выбрать заведомо смертельный ход.
Во время обучения щит не используется — сеть учится сама.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .expert import allowed_actions
from .game import ACTIONS, KEEP

if TYPE_CHECKING:
    from .agent import DQNAgent


def shield_mask(game) -> tuple[list[int], np.ndarray]:
    """Разрешённые щитом действия и маска для панели визуализации."""
    actions, _ = allowed_actions(game)
    mask = np.zeros(len(ACTIONS), dtype=bool)
    mask[actions] = True
    return actions, mask


def choose_action(agent: "DQNAgent", game, state: np.ndarray,
                  shield: bool = True) -> tuple[int, np.ndarray]:
    """Выбрать действие и маску, которую нужно показать на панели."""
    if not shield:
        return agent.act(state, greedy=True), np.ones(len(ACTIONS), dtype=bool)

    actions, mask = shield_mask(game)
    if not actions:
        return KEEP, mask
    if len(actions) == 1:
        # Щит оставил ровно один безопасный ход — сеть не нужна.
        return actions[0], mask

    values = agent.q_values(state)
    action = max(actions, key=lambda item: float(values[item]))
    return action, mask
