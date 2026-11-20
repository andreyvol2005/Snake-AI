"""Отрисовка игры в окне pygame на ассетах из оригинальной «Змейки».

Один экран 1000×1000, поле 18×18 по 50 px, как в исходной игре.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pygame

from .game import DOWN, LEFT, RIGHT, UP, SnakeGame

ASSETS = Path(__file__).resolve().parent.parent / "assets"
CELL = 50
OFFSET = (50, 55)          # как в оригинальной игре
GRID = 18
WINDOW = (1000, 1000)

# Оригинальный спрайт головы смотрит вправо; индекс поворота для transform.rotate.
ANGLE_INDEX = {RIGHT: 0, UP: 1, LEFT: 2, DOWN: 3}


class Renderer:
    def __init__(self, headless: bool = False):
        os_env = {"SDL_VIDEODRIVER": "dummy"} if headless else {}
        if headless:
            import os
            os.environ.update(os_env)
        pygame.init()
        self.headless = headless
        self.screen = pygame.display.set_mode(WINDOW)
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
        self.small_font = pygame.font.Font(str(ASSETS / "font.otf"), 28)

    # --------------------------------------------------------------- отрисовка
    def draw(self, game: SnakeGame, score: int, best: int, info: str = "") -> None:
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

        pygame.display.update()

    def close(self) -> None:
        pygame.quit()

    def pump(self) -> bool:
        """Обработать события; вернуть False, если окно закрыли."""
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