"""Обучение агента методом DQN (глубокое обучение с подкреплением).

Запуск:
    python train.py                 # обучение с periodically отрисовкой прогресса
    python train.py --episodes 1500 --state grid
    python train.py --no-plot       # не сохранять график обучения
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np

from snake_ai.agent import AgentConfig, DQNAgent
from snake_ai.game import GameConfig, SnakeGame
from snake_ai.state import encode, state_size

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"


def evaluate(agent: DQNAgent, episodes: int, cfg: GameConfig, kind: str, max_steps: int) -> tuple[float, float]:
    """Средний и максимальный счёт без случайных действий (greedy)."""
    scores = []
    for _ in range(episodes):
        game = SnakeGame(cfg, rng=random.Random())
        state = encode(game, kind)
        done = False
        while not done:
            action = agent.act(state, greedy=True)
            result = game.step(action)
            state = encode(game, kind)
            done = result.done or game.steps >= max_steps
        scores.append(game.score)
    return float(np.mean(scores)), float(np.max(scores))


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)

    game_cfg = GameConfig(max_steps=args.max_steps)
    agent_cfg = AgentConfig(
        state_kind=args.state,
        hidden=args.hidden,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch,
        target_sync=args.target_sync,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
        device=args.device,
    )
    size = state_size(args.state)
    agent = DQNAgent(size, agent_cfg)

    history_path = ROOT / "training_history.csv"
    best_avg = -1.0
    history = []
    t0 = time.time()

    print(f"Состояние: {args.state} ({size} признаков), устройство: {agent.device}")
    print(f"Эпизодов: {args.episodes}, максимум шагов за эпизод: {args.max_steps}")
    print("-" * 72)

    for episode in range(1, args.episodes + 1):
        game = SnakeGame(game_cfg, rng=random.Random(random.random()))
        state = encode(game, args.state)
        total_reward = 0.0
        done = False

        while not done:
            action = agent.act(state)
            result = game.step(action)
            reward = result.reward
            done = result.done or game.steps >= args.max_steps
            if game.steps >= args.max_steps and not result.done:
                reward -= 0.5  # затягивание партии штрафуется

            next_state = encode(game, args.state)
            agent.remember(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state
            total_reward += reward

        agent.decay_epsilon()  # ε снижаем раз в эпизод, а не каждый шаг

        history.append((episode, game.score, total_reward, agent.epsilon, agent.loss or 0.0))

        if episode % args.log_every == 0 or episode == 1:
            recent = history[-args.log_every:]
            avg_score = sum(h[1] for h in recent) / len(recent)
            avg_reward = sum(h[2] for h in recent) / len(recent)
            loss = agent.loss or 0.0
            elapsed = time.time() - t0
            print(f"Эпизод {episode:5d} | средний счёт {avg_score:5.1f} | "
                  f"средняя награда {avg_reward:7.2f} | ε {agent.epsilon:.3f} | "
                  f"loss {loss:6.3f} | {elapsed:5.1f} c")

        if episode % args.eval_every == 0:
            avg, best = evaluate(agent, args.eval_episodes, game_cfg, args.state, args.max_steps)
            print(f"  ↳ проверка без случайных ходов: средний {avg:.2f}, лучший {best:.0f}")
            if avg > best_avg:
                best_avg = avg
                agent.save(MODELS / f"best_{args.state}.pt")
                print(f"  ↳ новый лучший результат — модель сохранена в {MODELS / f'best_{args.state}.pt'}")

    agent.save(MODELS / f"last_{args.state}.pt")
    if not args.no_plot:
        save_plot(history, ROOT / f"training_{args.state}.png")

    history_path.write_text(
        "episode,score,reward,epsilon,loss\n"
        + "\n".join(f"{e},{s},{r:.3f},{eps:.4f},{l:.4f}" for e, s, r, eps, l in history),
        encoding="utf-8",
    )
    print("-" * 72)
    print(f"Готово. Модели: {MODELS}, история: {history_path}")


def save_plot(history, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib не установлен — график пропущен (pip install matplotlib)")
        return

    episodes = [h[0] for h in history]
    scores = [h[1] for h in history]
    window = max(1, min(50, len(scores)))
    smoothed = np.convolve(scores, np.ones(window) / window, mode="valid")

    plt.figure(figsize=(9, 4.5))
    plt.plot(episodes, scores, color="#b0bec5", linewidth=0.7, label="счёт за эпизод")
    plt.plot(episodes[window - 1:], smoothed, color="#2e7d32", linewidth=2, label="скользящее среднее")
    plt.xlabel("Эпизод")
    plt.ylabel("Счёт (съеденные яблоки)")
    plt.title("Обучение DQN-агента в «Змейке»")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Обучение DQN-агента в «Змейке»")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--state", choices=["vector", "grid"], default="vector")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--target-sync", type=int, default=1000)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--epsilon-decay", type=float, default=0.9995)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="cpu / cuda / mps / auto")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())