"""Отрисовка игры в окне pygame на ассетах из оригинальной «Змейки» + панель нейросети справа.

Один экран 1500×1000: слева поле 18×18 по 50 px (как в оригинале), справа —
чёрная панель с визуализацией DQN: нейроны, связи (веса) и прохождение сигналов.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pygame

from .game import DOWN, LEFT, RIGHT, UP, SnakeGame

ASSETS = Path(__file__).resolve().parent.parent / "assets"
CELL = 50
OFFSET = (50, 55)
GRID = 18
GAME_W = 1000
NET_W = 450
WINDOW = (GAME_W + NET_W, 1000)

ANGLE_INDEX = {RIGHT: 0, UP: 1, LEFT: 2, DOWN: 3}

# Цвета для сети
BG_NET = (12, 12, 18)
NEURON_BG = (40, 40, 55)
NEURON_BORDER = (180, 180, 200)
ACT_COLOR = (0, 255, 120)      # активность нейрона (зелёный)
WEIGHT_POS = (255, 80, 80)     # положительный вес (красный)
WEIGHT_NEG = (80, 140, 255)    # отрицательный вес (синий)
CHOSEN_COLOR = (255, 220, 60)  # выбранное действие (жёлтый)
TEXT_COLOR = (220, 220, 230)


class Renderer:
    def __init__(self, headless: bool = False, show_network: bool = True):
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        self.headless = headless
        self.show_network = show_network
        size = WINDOW if show_network else (GAME_W, 1000)
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption("Змейка с ИИ")
        try:
            pygame.display.set_icon(pygame.image.load(str(ASSETS / "ava.png")))
        except pygame.error:
            pass

        self.bg = pygame.image.load(str(ASSETS / "bg.png")).convert_alpha()
        self.body = pygame.image.load(str(ASSETS / "snake_body.png")).convert_alpha()
        self.face = pygame.image.load(str(ASSETS / "snake_face.png")).convert_alpha()
        self.death = pygame.image.load(str(ASSETS / "death.png")).convert_alpha()
        self.apple = pygame.image.load(str(ASSETS / "apple.png")).convert_alpha()
        self.replay = pygame.image.load(str(ASSETS / "replay.png")).convert_alpha()
        self.font = pygame.font.Font(str(ASSETS / "font.otf"), 40)
        self.small_font = pygame.font.Font(str(ASSETS / "font.otf"), 24)
        self.tiny_font = pygame.font.Font(str(ASSETS / "font.otf"), 16)
        self.tinier_font = pygame.font.Font(str(ASSETS / "font.otf"), 12)

    # --------------------------------------------------------------- отрисовка игры
    def draw_game(self, game: SnakeGame, score: int, best: int, info: str = "") -> None:
        self.screen.blit(self.bg, (0, 0))

        score_text = self.font.render(f"Счёт: {score}", True, "white")
        record_text = self.font.render(f"Рекорд: {best}", True, "white")
        self.screen.blit(score_text, (50, 1))
        self.screen.blit(record_text, (300, 1))
        self.screen.blit(self.replay, (900, 1))
        if info:
            self.screen.blit(self.small_font.render(info, True, "white"), (560, 8))

        ox, oy = OFFSET
        if game.food is not None:
            fx, fy = game.food
            rect = self.apple.get_rect(topleft=(ox + fx * CELL, oy + fy * CELL))
            self.screen.blit(self.apple, rect)

        for i, (x, y) in enumerate(game.snake):
            img = self.death if game.done and i == 0 else self.body
            if i == 0 and not game.done:
                img = pygame.transform.rotate(self.face, ANGLE_INDEX[game.direction] * 90)
            self.screen.blit(img, (ox + x * CELL, oy + y * CELL))

    # ------------------------------------------------------- отрисовка нейросети
    def draw_network(
        self,
        state_vector: np.ndarray,
        activations: List[np.ndarray],
        weights: List[Tuple[np.ndarray, np.ndarray]],
        chosen_action: int,
    ) -> None:
        """Чёрная панель справа: нейроны, веса, активность, выбранное действие."""
        if not self.show_network:
            return

        panel_x = GAME_W + 20

        # Заливка панели
        pygame.draw.rect(self.screen, BG_NET, (GAME_W, 0, NET_W, 1000))
        title = self.tiny_font.render("Нейросеть DQN", True, TEXT_COLOR)
        self.screen.blit(title, (panel_x, 14))

        # Слои: вход (11), скрытые 1 и 2 (показываем до 16 нейронов каждый), выход (3)
        input_vals = state_vector
        hid1 = activations[0] if len(activations) > 0 else np.array([])
        hid2 = activations[1] if len(activations) > 1 else np.array([])

        layers_vals = [input_vals, hid1, hid2]
        # Ограничиваем скрытые слои 16 нейронами для читаемости
        layers_vals = [
            layers_vals[0],
            layers_vals[1][:16] if layers_vals[1].size else layers_vals[1],
            layers_vals[2][:16] if layers_vals[2].size else layers_vals[2],
        ]

        layer_xs = [panel_x + 60, panel_x + 150, panel_x + 240, panel_x + 330]

        layer_sizes = [len(layers_vals[0]), len(layers_vals[1]), len(layers_vals[2]), 3]

        # Нормализация активаций для яркости. Скрытые слои нормируем по
        # максимуму в слое, иначе при малых значениях рисунок почти не виден.
        def norm(vals: np.ndarray, per_layer: bool = False) -> np.ndarray:
            if vals.size == 0:
                return vals
            v = np.tanh(vals)
            if per_layer:
                peak = float(np.max(np.abs(v)))
                if peak > 1e-6:
                    v = v / peak
            return (v + 1) / 2  # [0, 1]

        input_bright = norm(input_vals)
        hid1_bright = norm(hid1, per_layer=True)
        hid2_bright = norm(hid2, per_layer=True)

        layer_brights = [input_bright, hid1_bright, hid2_bright, np.array([0.5, 0.5, 0.5])]

        # Связи между слоями (веса)
        w_pairs = weights  # [(w1,b1), (w2,b2), (w3,b3)]
        # w1: (hid1, input), w2: (hid2, hid1), w3: (out, hid2)
        connections = [
            (w_pairs[0][0] if len(w_pairs) > 0 else None, 0, 1),
            (w_pairs[1][0] if len(w_pairs) > 1 else None, 1, 2),
            (w_pairs[2][0] if len(w_pairs) > 2 else None, 2, 3),
        ]

        # Координаты нейронов
        def layer_coords(layer_idx: int) -> List[Tuple[int, int]]:
            size = layer_sizes[layer_idx]
            if size == 0:
                return []
            x = layer_xs[layer_idx]
            total_h = 700
            start_y = 120
            step = total_h / max(1, size)
            return [(x, int(start_y + i * step + step / 2)) for i in range(size)]

        coords = [layer_coords(i) for i in range(4)]

        # Рисуем связи (от заднего к переднему, чтобы нейроны были поверх).
        # Линии кладём на общий прозрачный слой — так не обрезаются наклонные.
        connections_layer = pygame.Surface((NET_W, 1000), pygame.SRCALPHA)
        for w_mat, li, lj in connections:
            if w_mat is None:
                continue
            src_bright = layer_brights[li]
            for i, (x1, y1) in enumerate(coords[li]):
                if i >= w_mat.shape[1]:
                    continue
                # Сигнал от нейрона-источника: его текущая активация.
                act = float(src_bright[i]) if i < len(src_bright) else 0.5
                for j, (x2, y2) in enumerate(coords[lj]):
                    if j >= w_mat.shape[0]:
                        continue
                    w = float(w_mat[j, i])
                    if abs(w) < 0.02:
                        continue
                    color = WEIGHT_POS if w > 0 else WEIGHT_NEG
                    thickness = max(1, int(4 * abs(w)))
                    # Чем сильнее активация источника и больше вес,
                    # тем ярче связь — видно, как сигнал идёт по сети.
                    alpha = int(30 + 215 * min(1.0, abs(w)) * (0.2 + 0.8 * act))
                    pygame.draw.line(
                        connections_layer,
                        (*color, alpha),
                        (x1 - GAME_W, y1),
                        (x2 - GAME_W, y2),
                        thickness,
                    )
        self.screen.blit(connections_layer, (GAME_W, 0))

        # Рисуем нейроны
        for li in range(4):
            if not coords[li]:
                continue
            brights = layer_brights[li]
            for j, (x, y) in enumerate(coords[li]):
                r = 7 if li in (0, 3) else 6
                # Яркость от активации
                b = float(brights[j]) if j < len(brights) else 0.0
                base = 40 + int(150 * b)
                color = (base, base, base + 30)
                if li == 3:
                    # Выходной слой: подсвечиваем выбранное действие
                    if j == chosen_action:
                        color = CHOSEN_COLOR
                    else:
                        color = (base, base, base)
                pygame.draw.circle(self.screen, NEURON_BG, (x, y), r)
                pygame.draw.circle(self.screen, color, (x, y), max(1, r - 2))
                pygame.draw.circle(self.screen, NEURON_BORDER, (x, y), r, 1)

                # Подписи для входных нейронов (первые несколько)
                if li == 0 and j < 6:
                    lbl = self.tinier_font.render(f"x{j}", True, (110, 110, 125))
                    self.screen.blit(lbl, (x - 22, y - 5))

        # Подписи слоёв — внизу панели, коротко, в одну строку.
        short_names = ["вход", "скр. 1", "скр. 2", "выход"]
        for li, name in enumerate(short_names):
            if not coords[li]:
                continue
            x = layer_xs[li]
            lbl = self.tinier_font.render(name, True, TEXT_COLOR)
            self.screen.blit(lbl, (x - lbl.get_width() // 2, 950))

        # Легенда действий
        legend = ["прямо", "влево", "вправо"]
        for j in range(3):
            x, y = coords[3][j] if coords[3] else (panel_x + 330, 120 + j * 100)
            lbl = self.tiny_font.render(legend[j], True, CHOSEN_COLOR if j == chosen_action else (100, 100, 110))
            self.screen.blit(lbl, (x + 12, y - 8))

    # --------------------------------------------------------------- общий вызов
    def draw(
        self,
        game: SnakeGame,
        score: int,
        best: int,
        info: str = "",
        state_vector: Optional[np.ndarray] = None,
        activations: Optional[List[np.ndarray]] = None,
        weights: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        chosen_action: Optional[int] = None,
    ) -> None:
        self.draw_game(game, score, best, info)
        if (
            self.show_network
            and state_vector is not None
            and activations is not None
            and weights is not None
            and chosen_action is not None
        ):
            self.draw_network(state_vector, activations, weights, chosen_action)
        pygame.display.update()

    def close(self) -> None:
        pygame.quit()

    def pump(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

    def wait_click(self) -> bool:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    return True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                    return True

    @staticmethod
    def cell_to_px(point: Tuple[int, int]) -> Tuple[int, int]:
        ox, oy = OFFSET
        return ox + point[0] * CELL, oy + point[1] * CELL