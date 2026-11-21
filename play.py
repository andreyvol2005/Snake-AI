"""Просмотр игры обученного агента в окне pygame.

Запуск:
    python play.py                 # модель по умолчанию, обычная скорость
    python play.py --model models/best_vector.pt --speed 40
    python play.py --human         # играть самому, клавиши WASD / стрелки
    python play.py --episodes 0    # бесконечная игра до закрытия окна
    python play.py --no-network    # без панели нейросети (узкое окно)
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pygame

from snake_ai.agent import DQNAgent
from snake_ai.game import (
    DOWN,
    KEEP,
    LEFT,
    RIGHT,
    TURN_LEFT,
    TURN_RIGHT,
    UP,
    GameConfig,
    SnakeGame,
)
from snake_ai.render import Renderer
from snake_ai.state import encode, state_size

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"

BEST_FILE = ROOT / "record"

# Клавиши -> абсолютное направление; поворот только на 90°, разворот запрещён.
KEY_TO_DIRECTION = {
    pygame.K_w: UP, pygame.K_UP: UP,
    pygame.K_d: RIGHT, pygame.K_RIGHT: RIGHT,
    pygame.K_s: DOWN, pygame.K_DOWN: DOWN,
    pygame.K_a: LEFT, pygame.K_LEFT: LEFT,
}


def read_best() -> int:
    try:
        return int(BEST_FILE.read_text().strip())
    except (OSError, ValueError):
        return 0


def write_best(value: int) -> None:
    BEST_FILE.write_text(str(value), encoding="utf-8")


def load_agent(path: Path, kind: str) -> DQNAgent:
    if not path.exists():
        raise SystemExit(f"Модель не найдена: {path}. Сначала запустите train.py")
    agent = DQNAgent.load(path, state_size(kind))
    agent.epsilon = 0.0  # играем без случайных движений
    return agent


def run_agent(args: argparse.Namespace) -> None:
    cfg = GameConfig(max_steps=args.max_steps)
    agent = load_agent(args.model, args.state)
    renderer = Renderer(show_network=not args.no_network)
    best = read_best()
    clock = pygame.time.Clock()
    games = 0
    total = 0
    # Веса не меняются во время игры — берём один раз.
    weights = agent.get_weights() if not args.no_network else None

    try:
        while args.episodes == 0 or games < args.episodes:
            if not renderer.pump():
                break
            game = SnakeGame(cfg, rng=random.Random())
            state = encode(game, args.state)
            activations = None
            action = KEEP
            done = False

            while not done:
                if not renderer.pump():
                    break

                if args.no_network:
                    action = agent.act(state, greedy=True)
                else:
                    action, activations = agent.act_with_activations(state, greedy=True)

                result = game.step(action)
                state = encode(game, args.state)
                done = result.done
                if game.score > best:
                    best = game.score
                    write_best(best)

                renderer.draw(
                    game, game.score, best, f"шаг {game.steps}",
                    state_vector=state,
                    activations=activations,
                    weights=weights,
                    chosen_action=action,
                )
                clock.tick(args.speed)

            games += 1
            total += game.score
            print(f"Игра {games}: счёт {game.score}, средний {total / games:.2f}")
            if not renderer.wait_click():
                break
    finally:
        renderer.close()


def run_human(args: argparse.Namespace) -> None:
    cfg = GameConfig(max_steps=args.max_steps)
    renderer = Renderer(show_network=False)
    best = read_best()
    clock = pygame.time.Clock()

    def apply_key(key: int) -> None:
        """Повернуть к желаемому направлению, если это не разворот на 180°."""
        target = KEY_TO_DIRECTION.get(key)
        if target is None:
            return
        diff = (target - game.direction) % 4
        if diff == 1:
            game.direction = TURN_RIGHT[game.direction]
        elif diff == 3:
            game.direction = TURN_LEFT[game.direction]
        # diff == 2 — разворот назад, игнорируем

    try:
        while True:
            if not renderer.pump():
                break
            game = SnakeGame(cfg, rng=random.Random())
            done = False
            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        renderer.close()
                        return
                    if event.type == pygame.KEYDOWN:
                        apply_key(event.key)

                result = game.step(KEEP)
                done = result.done
                if game.score > best:
                    best = game.score
                    write_best(best)
                renderer.draw(game, game.score, best, "ты")
                clock.tick(args.speed)

            print(f"Игра окончена: счёт {game.score}, рекорд {best}")
            if not renderer.wait_click():
                break
    finally:
        renderer.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Игра змейки с ИИ или человеком")
    p.add_argument("--model", type=Path, default=MODELS / "best_vector.pt")
    p.add_argument("--state", choices=["vector", "vector24", "grid"], default="vector")
    p.add_argument("--speed", type=int, default=30, help="шагов в секунду")
    p.add_argument("--episodes", type=int, default=3, help="0 — играть бесконечно")
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--human", action="store_true", help="играть самому")
    p.add_argument("--no-network", action="store_true", help="не показывать панель нейросети")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.human:
        run_human(args)
    else:
        run_agent(args)