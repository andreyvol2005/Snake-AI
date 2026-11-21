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

from .agent import NetworkSnapshot
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
WEIGHT_POS = (80, 140, 255)    # положительный вес (синий)
WEIGHT_NEG = (255, 80, 80)     # отрицательный вес (красный)
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
        self.micro_font = pygame.font.Font(str(ASSETS / "font.otf"), 10)

    # --------------------------------------------------------------- отрисовка игры
    def draw_game(self, game: SnakeGame, score: int, best: int, info: str = "") -> None:
        self.screen.blit(self.bg, (0, 0))

        score_text = self.font.render(f"Счёт: {score}", True, "white")
        record_text = self.font.render(f"Рекорд: {best}", True, "white")
        self.screen.blit(score_text, (50, 1))
        self.screen.blit(record_text, (300, 1))
        self.screen.blit(self.replay, (900, 1))
        if info:
            self.screen.blit(self.small_font.render(info, True, "white"), (630, 8))

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
    @staticmethod
    def _bright(vals: np.ndarray, floor: float = 0.06) -> np.ndarray:
        """Яркость [0,1] по абсолютной величине, нормированная на максимум слоя."""
        v = np.abs(np.asarray(vals, dtype=np.float32)).ravel()
        if v.size == 0:
            return v
        peak = float(v.max())
        if peak > 1e-9:
            v = v / peak
        return np.clip(floor + (1.0 - floor) * v, 0.0, 1.0)

    @staticmethod
    def _mix(a: Tuple[int, int, int], b: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
        t = max(0.0, min(1.0, t))
        return (int(a[0] + (b[0] - a[0]) * t),
                int(a[1] + (b[1] - a[1]) * t),
                int(a[2] + (b[2] - a[2]) * t))

    def _draw_input_map(self, snap: NetworkSnapshot, x: int, y: int, cell: int = 5) -> None:
        board = np.asarray(snap.input_values)
        if board.ndim != 2:
            return
        head = np.asarray(snap.head_map)
        h, w = board.shape
        pygame.draw.rect(self.screen, (55, 55, 70), (x - 2, y - 2, w * cell + 3, h * cell + 3), 1)
        for yy in range(h):
            for xx in range(w):
                value = float(board[yy, xx])
                if head.size and float(head[yy, xx]) > 0.5:
                    color = (240, 240, 255)
                elif value > 0.5:
                    color = (80, 230, 100)
                elif value < -0.5:
                    color = (55, 105, 205)
                else:
                    color = (18, 18, 26)
                pygame.draw.rect(self.screen, color, (x + xx * cell, y + yy * cell, cell - 1, cell - 1))

    def _draw_conv_maps(self, acts: np.ndarray, x: int, y: int, scale: int,
                        label: str, cols: int = 8, count: int = 16) -> None:
        """Карты активаций одной свёртки.

        Размер карт задаётся явно: у первой свёртки они большие (18×18),
        у второй и третьей — мелкие, чтобы все 32 фильтра поместились.
        """
        if acts is None or acts.size == 0:
            return
        lbl = self.micro_font.render(label, True, (150, 150, 170))
        self.screen.blit(lbl, (x, y - 13))
        gap = 3
        n_maps = min(count, int(acts.shape[0]))
        size = int(np.asarray(acts[0]).shape[0])
        for k in range(n_maps):
            fmap = np.asarray(acts[k], dtype=np.float32)
            peak = float(fmap.max())
            if peak > 1e-9:
                fmap = fmap / peak
            gx = x + (k % cols) * (size * scale + gap)
            gy = y + (k // cols) * (size * scale + gap)
            for yy in range(fmap.shape[0]):
                for xx in range(fmap.shape[1]):
                    b = float(np.clip(fmap[yy, xx], 0.0, 1.0)) ** 0.65
                    color = (int(20 + 225 * b), int(25 + 225 * b), int(35 + 220 * b))
                    pygame.draw.rect(self.screen, color,
                                     (gx + xx * scale, gy + yy * scale, scale, scale))
            pygame.draw.rect(self.screen, (70, 70, 95),
                             (gx - 1, gy - 1, size * scale + 1, size * scale + 1), 1)

    def draw_network(self, snap: NetworkSnapshot) -> None:
        """Чёрная панель справа: карта входа, свёртки, нейроны, веса, сигнал."""
        if not self.show_network:
            return

        panel_x = GAME_W + 15

        pygame.draw.rect(self.screen, BG_NET, (GAME_W, 0, NET_W, 1000))
        title = self.tiny_font.render("Нейросеть DQN + BFS-щит", True, TEXT_COLOR)
        self.screen.blit(title, (panel_x, 8))

        n_hidden = len(snap.hidden_activations)
        if snap.arch == "cnn":
            subtitle = f"карта 18×18 · свёртки · {n_hidden} скрытых слоя"
        else:
            subtitle = f"{len(np.ravel(snap.input_values))} признаков · {n_hidden} скрытых слоя"
        self.screen.blit(self.micro_font.render(subtitle, True, (130, 130, 150)), (panel_x, 27))

        # ---------------------------------------------------------- зрение сети
        if snap.arch == "cnn":
            self.screen.blit(self.micro_font.render("вход 18×18", True, (150, 150, 170)),
                             (panel_x, 43))
            self._draw_input_map(snap, panel_x, 56, cell=6)

            # Легенда карты.
            legend_items = [
                ((80, 230, 100), "яблоко"),
                ((55, 105, 205), "змея"),
                ((240, 240, 255), "голова"),
                ((45, 45, 55), "пусто"),
            ]
            for i, (color, name) in enumerate(legend_items):
                ly = 172 + i * 14
                pygame.draw.rect(self.screen, color, (panel_x, ly, 8, 8))
                self.screen.blit(self.micro_font.render(name, True, (135, 135, 150)),
                                 (panel_x + 12, ly - 1))

            conv = snap.conv_activations
            if len(conv) > 0:
                self._draw_conv_maps(
                    conv[0], panel_x, 56, 2,
                    f"свёртка 1 · {min(16, conv[0].shape[0])}/{conv[0].shape[0]}",
                    cols=8, count=16,
                )
            if len(conv) > 1:
                self._draw_conv_maps(
                    conv[1], panel_x, 162, 3,
                    f"свёртка 2 · {min(32, conv[1].shape[0])}/{conv[1].shape[0]}",
                    cols=16, count=32,
                )
            if len(conv) > 2:
                self._draw_conv_maps(
                    conv[2], panel_x, 270, 3,
                    f"свёртка 3 · {min(32, conv[2].shape[0])}/{conv[2].shape[0]}",
                    cols=16, count=32,
                )
            self.screen.blit(
                self.micro_font.render("яркость карты = активация фильтра", True, (95, 95, 110)),
                (panel_x, 340),
            )
            pygame.draw.line(self.screen, (45, 45, 58), (panel_x, 356), (panel_x + 425, 356))
            top, bottom = 382, 895
        else:
            top, bottom = 95, 895

        # -------------------------------------------------------- решение сети
        if snap.arch == "cnn":
            in_vals = snap.feature_values
            feature_idx = np.linspace(0, in_vals.size - 1, 16).astype(int) if in_vals.size else np.array([], int)
            in_vals = in_vals[feature_idx] if in_vals.size else np.zeros(16, np.float32)
            layer_values = [in_vals]
        else:
            feature_idx = None
            layer_values = [np.asarray(snap.input_values).ravel()]

        shown = 16
        layer_values += [np.asarray(h).ravel()[:shown] for h in snap.hidden_activations]
        layer_values.append(np.asarray(snap.output_values).ravel())

        n_cols = len(layer_values)
        xs = np.linspace(panel_x + 35, panel_x + 360, n_cols).astype(int)

        def coords_for(idx: int) -> List[Tuple[int, int]]:
            vals = layer_values[idx]
            if vals.size == 0:
                return []
            x = int(xs[idx])
            if idx == n_cols - 1:  # выход: три оценки по центру
                mid = (top + bottom) / 2
                return [(x, int(mid + (i - (len(vals) - 1) / 2) * 110)) for i in range(len(vals))]
            step = (bottom - top) / len(vals)
            return [(x, int(top + step / 2 + i * step)) for i in range(len(vals))]

        coords = [coords_for(i) for i in range(n_cols)]

        input_bright = self._bright(layer_values[0])
        hidden_brights = [self._bright(v) for v in layer_values[1:-1]]
        out_vals = layer_values[-1]
        if out_vals.size > 1:
            span = float(out_vals.max() - out_vals.min())
            out_bright = (out_vals - out_vals.min()) / span if span > 1e-9 else np.full(out_vals.shape, 0.5)
            out_bright = 0.15 + 0.85 * out_bright
        else:
            out_bright = np.array([0.5])
        layer_brights = [input_bright, *hidden_brights, out_bright]

        # Матрицы весов для показанных нейронов.
        mats: List[np.ndarray] = []
        for k, W in enumerate(snap.dense_weights):
            if k == 0:
                if snap.arch == "cnn":
                    if feature_idx is None or feature_idx.size == 0:
                        continue
                    mats.append(W[:shown][:, feature_idx])
                else:
                    mats.append(W[:shown, :len(layer_values[0])])
            elif k == len(snap.dense_weights) - 1:
                mats.append(W[:, :shown])
            else:
                mats.append(W[:shown, :shown])

        # Связи рисуем на прозрачном слое, нейроны — поверх.
        connections_layer = pygame.Surface((NET_W, 1000), pygame.SRCALPHA)
        for k, w_mat in enumerate(mats):
            if k + 1 >= len(coords):
                break
            src_bright = layer_brights[k]
            for i, (x1, y1) in enumerate(coords[k]):
                if i >= w_mat.shape[1]:
                    continue
                act = float(src_bright[i]) if i < len(src_bright) else 0.5
                for j, (x2, y2) in enumerate(coords[k + 1]):
                    if j >= w_mat.shape[0]:
                        continue
                    w = float(w_mat[j, i])
                    if abs(w) < 0.02:
                        continue
                    color = WEIGHT_POS if w > 0 else WEIGHT_NEG
                    thickness = max(1, min(4, int(1 + 6 * abs(w))))
                    alpha = int(25 + 210 * min(1.0, abs(w) * 2.5) * (0.15 + 0.85 * act))
                    pygame.draw.line(
                        connections_layer, (*color, alpha),
                        (x1 - GAME_W, y1), (x2 - GAME_W, y2), thickness,
                    )
        self.screen.blit(connections_layer, (GAME_W, 0))

        # Нейроны.
        for li in range(n_cols):
            brights = layer_brights[li]
            is_out = li == n_cols - 1
            for j, (x, y) in enumerate(coords[li]):
                r = 8 if is_out else (6 if li else 5)
                b = float(brights[j]) if j < len(brights) else 0.0
                base = 35 + int(160 * b)
                color: Tuple[int, int, int] = (base, base, min(255, base + 25))
                allowed = (j >= len(snap.action_mask) or bool(snap.action_mask[j]))
                if is_out:
                    if not allowed:
                        color = (52, 52, 62)
                    else:
                        color = CHOSEN_COLOR if j == snap.chosen_action else (base, base, base)
                pygame.draw.circle(self.screen, NEURON_BG, (x, y), r)
                pygame.draw.circle(self.screen, color, (x, y), max(1, r - 2))
                pygame.draw.circle(self.screen, NEURON_BORDER, (x, y), r, 1)
                if is_out and not allowed:
                    # Зачёркнутый выход: щит запретил этот ход.
                    pygame.draw.line(self.screen, (95, 95, 110),
                                     (x - 4, y - 4), (x + 4, y + 4), 1)
                    pygame.draw.line(self.screen, (95, 95, 110),
                                     (x - 4, y + 4), (x + 4, y - 4), 1)

                if li == 0 and snap.arch == "mlp" and j < 11:
                    lbl = self.micro_font.render(f"x{j}", True, (105, 105, 120))
                    self.screen.blit(lbl, (x - 20, y - 5))

            if is_out:
                names = ["прямо", "влево", "вправо"]
                for j, (x, y) in enumerate(coords[li]):
                    if j >= len(names):
                        break
                    chosen = j == snap.chosen_action
                    allowed = (j >= len(snap.action_mask) or bool(snap.action_mask[j]))
                    if chosen:
                        name_color = CHOSEN_COLOR
                    elif allowed:
                        name_color = (105, 105, 118)
                    else:
                        name_color = (70, 70, 82)
                    lbl = self.tinier_font.render(names[j], True, name_color)
                    self.screen.blit(lbl, (x + 12, y - 12))
                    q = float(layer_values[-1][j]) if j < layer_values[-1].size else 0.0
                    if chosen:
                        q_color = CHOSEN_COLOR
                    elif allowed:
                        q_color = (85, 85, 98)
                    else:
                        q_color = (62, 62, 74)
                    qlbl = self.micro_font.render(f"Q {q:+.2f}", True, q_color)
                    self.screen.blit(qlbl, (x + 13, y + 1))

        # Подписи слоёв.
        if snap.arch == "cnn":
            names = ["признаки"] + [f"скр. {i + 1}" for i in range(n_hidden)] + ["выход"]
        else:
            names = ["вход"] + [f"скр. {i + 1}" for i in range(n_hidden)] + ["выход"]
        for li, name in enumerate(names):
            if li >= len(coords) or not coords[li]:
                continue
            lbl = self.tinier_font.render(name, True, TEXT_COLOR)
            self.screen.blit(lbl, (int(xs[li]) - lbl.get_width() // 2, 928))

        # Легенда весов.
        pygame.draw.line(self.screen, WEIGHT_POS, (panel_x, 958), (panel_x + 20, 958), 2)
        self.screen.blit(self.micro_font.render("вес +", True, (140, 140, 155)), (panel_x + 25, 952))
        pygame.draw.line(self.screen, WEIGHT_NEG, (panel_x + 85, 958), (panel_x + 105, 958), 2)
        self.screen.blit(self.micro_font.render("вес −", True, (140, 140, 155)), (panel_x + 110, 952))
        self.screen.blit(self.micro_font.render("яркость связи = сигнал", True, (100, 100, 115)),
                         (panel_x + 180, 952))
        self.screen.blit(
            self.micro_font.render("зачёркнутый выход = ход отсечён щитом", True, (85, 85, 100)),
            (panel_x, 970),
        )

    # --------------------------------------------------------------- общий вызов
    def draw(
        self,
        game: SnakeGame,
        score: int,
        best: int,
        info: str = "",
        snapshot: Optional[NetworkSnapshot] = None,
    ) -> None:
        self.draw_game(game, score, best, info)
        if self.show_network and snapshot is not None:
            self.draw_network(snapshot)
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