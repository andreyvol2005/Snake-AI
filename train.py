"""Обучение агента методом DQN (глубокое обучение с подкреплением).

Запуск:
    python train.py                        # карта поля 18×18 + свёрточная сеть
    python train.py --state vector         # быстрый вариант на 11 признаках
    python train.py --episodes 12000       # дольше — выше потолок счёта
    python train.py --no-plot              # не рисовать график обучения
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np

from snake_ai.agent import AgentConfig, DQNAgent, ReplayBuffer
from snake_ai.expert import expert_action
from snake_ai.game import GameConfig, SnakeGame
from snake_ai.policy import choose_action
from snake_ai.state import encode, state_shape

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"


def evaluate(agent: DQNAgent, episodes: int, cfg: GameConfig, kind: str,
             shield: bool = False) -> dict:
    """Оценка без случайных действий: средний, медиана, максимум, доля 40+.

    ``shield=True`` включает BFS-страховку, с которой агент играет в
    ``play.py``; ``shield=False`` показывает чистую политику сети.
    """
    agent.sync_actor()
    scores = []
    for _ in range(episodes):
        game = SnakeGame(cfg, rng=random.Random())
        state = encode(game, kind)
        done = False
        while not done:
            action, _ = choose_action(agent, game, state, shield=shield)
            result = game.step(action)
            state = encode(game, kind)
            done = result.done
        scores.append(game.score)
    arr = np.asarray(scores, dtype=np.float32)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "max": int(arr.max()),
        "over40": float((arr >= 40).mean() * 100.0),
        "scores": scores,
    }


def make_demonstrations(games: int, cfg: GameConfig, kind: str,
                        capacity: int, seed: int = 10_000) -> ReplayBuffer:
    """Прогнать BFS-учителя и сохранить его переходы для DQfD."""
    demo = ReplayBuffer(capacity, state_shape(kind))
    for episode in range(games):
        game = SnakeGame(cfg, rng=random.Random(seed + episode))
        state = encode(game, kind)
        done = False
        while not done:
            action = expert_action(game)
            result = game.step(action)
            next_state = encode(game, kind)
            demo.add(state, action, result.reward, next_state, result.done)
            state = next_state
            done = result.done
    return demo


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)

    game_cfg = GameConfig(max_steps=args.max_steps)
    agent_cfg = AgentConfig(
        state_kind=args.state,
        arch=args.arch,
        hidden=args.hidden,
        hidden_layers=args.hidden_layers,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch,
        buffer_size=args.buffer_size,
        target_sync=args.target_sync,
        learn_start=args.learn_start,
        epsilon_start=args.epsilon_start if args.epsilon_start is not None else 1.0,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        demo_ratio=args.demo_ratio,
        aux_coef=args.aux_coef,
        expert_margin=args.expert_margin,
        seed=args.seed,
        device=args.device,
    )
    shape = state_shape(args.state)
    if args.load:
        agent = DQNAgent.load(args.load, shape)
        if tuple(agent.state_shape) != tuple(shape):
            raise SystemExit(
                f"Чекпоинт {args.load} хранит состояние {agent.state_shape}, "
                f"а --state {args.state} даёт {shape}. "
                "Старые 4-канальные grid-модели нужно переобучить."
            )
        agent.cfg.lr = args.lr
        agent.cfg.gamma = args.gamma
        agent.cfg.batch_size = args.batch
        agent.cfg.target_sync = args.target_sync
        agent.cfg.buffer_size = args.buffer_size
        agent.cfg.demo_ratio = args.demo_ratio
        agent.cfg.aux_coef = args.aux_coef
        agent.cfg.expert_margin = args.expert_margin
        for group in agent.optimizer.param_groups:
            group["lr"] = args.lr
        agent.set_buffer_size(args.buffer_size)
        if args.epsilon_start is not None:
            agent.epsilon = args.epsilon_start
        kind = agent.cfg.state_kind
        print(f"Продолжаем обучение с {args.load} (состояние: {kind}, {agent.cfg.arch})")
    else:
        agent = DQNAgent(shape, agent_cfg)
        kind = args.state

    if args.demo_games > 0:
        t_demo = time.time()
        demo = make_demonstrations(args.demo_games, game_cfg, kind,
                                   args.demo_size or args.buffer_size)
        agent.set_demo_buffer(demo)
        print(f"Демонстрации учителя: {len(demo)} переходов из {args.demo_games} партий "
              f"за {time.time() - t_demo:.1f} c")

    history_path = ROOT / f"training_history_{kind}.csv"
    history = []
    t0 = time.time()
    total_steps = 0

    print(f"Состояние: {kind} {shape}, архитектура: {agent.cfg.arch}, устройство: {agent.device}")
    print(f"Эпизодов: {args.episodes}, максимум шагов на партию: {args.max_steps}")
    print(f"Скрытых слоёв: {agent.cfg.hidden_layers} × {agent.cfg.hidden}; "
          f"ε: {agent.epsilon:.2f} → {args.epsilon_min}, затухание {args.epsilon_decay}")
    print("-" * 72)

    # При продолжении обучения стартуем от текущего качества загруженной
    # модели — иначе плохой прогон перезапишет хороший чекпоинт.
    if args.load:
        base = evaluate(agent, args.eval_episodes, game_cfg, kind, shield=False)
        base_shield = evaluate(agent, args.eval_episodes, game_cfg, kind, shield=True)
        best_pure = base["mean"]
        best_shield = base_shield["mean"]
        print(f"Исходное качество: сеть {base['mean']:.2f} (медиана {base['median']:.1f}, "
              f"максимум {base['max']}, 40+ {base['over40']:.0f}%) | "
              f"со щитом {base_shield['mean']:.2f} (40+ {base_shield['over40']:.0f}%)")
    else:
        best_pure = -1.0
        best_shield = -1.0

    for episode in range(1, args.episodes + 1):
        game = SnakeGame(game_cfg, rng=random.Random(random.random()))
        state = encode(game, kind)
        total_reward = 0.0
        done = False
        frac = (episode - 1) / max(1, args.episodes - 1)
        expert_prob = args.expert_prob + (args.expert_prob_end - args.expert_prob) * frac

        while not done:
            # Смешанное поведение: часть ходов подсказывает учитель, часть
            # выбирает сеть. Так агент видит длинные змейки и учится
            # выходить из положений, в которые сам же попадает.
            if expert_prob > 0.0 and random.random() < expert_prob:
                action = expert_action(game)
            else:
                action = agent.act(state)
            result = game.step(action)
            reward = result.reward
            done = result.done
            total_steps += 1

            next_state = encode(game, kind)
            agent.remember(state, action, reward, next_state, done)
            if total_steps % args.train_every == 0:
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
                  f"средняя награда {avg_reward:8.2f} | ε {agent.epsilon:.3f} | "
                  f"loss {loss:6.3f} | {elapsed:6.1f} c", flush=True)

        if episode % args.eval_every == 0:
            ev = evaluate(agent, args.eval_episodes, game_cfg, kind, shield=False)
            ev_shield = evaluate(agent, args.eval_episodes, game_cfg, kind, shield=True)
            print(f"  ↳ проверка: сеть {ev['mean']:.2f} "
                  f"(медиана {ev['median']:.1f}, лучший {ev['max']}, "
                  f"40+ {ev['over40']:.0f}%) | со щитом {ev_shield['mean']:.2f} "
                  f"(медиана {ev_shield['median']:.1f}, лучший {ev_shield['max']}, "
                  f"40+ {ev_shield['over40']:.0f}%)", flush=True)
            # Сначала смотрим на игру со щитом, затем — на чистую сеть.
            if (ev_shield["mean"], ev["mean"]) > (best_shield, best_pure):
                best_shield = ev_shield["mean"]
                best_pure = ev["mean"]
                model_path = MODELS / f"best_{kind}.pt"
                agent.save(model_path)
                print(f"  ↳ новый лучший результат — модель сохранена в {model_path}")

    agent.save(MODELS / f"last_{kind}.pt")
    if not args.no_plot:
        save_plot(history, ROOT / f"training_{kind}.png")

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
    p.add_argument("--episodes", type=int, default=1500)
    p.add_argument("--state", choices=["vector", "vector24", "grid"], default="grid")
    p.add_argument("--arch", choices=["auto", "mlp", "cnn"], default="auto")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--hidden-layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--buffer-size", type=int, default=30_000)
    p.add_argument("--target-sync", type=int, default=2000)
    p.add_argument("--learn-start", type=int, default=500)
    p.add_argument("--train-every", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=800)
    p.add_argument("--epsilon-start", type=float, default=None,
                   help="переопределить начальный ε (по умолчанию 1.0 или из чекпоинта)")
    p.add_argument("--epsilon-decay", type=float, default=0.9975)
    p.add_argument("--epsilon-min", type=float, default=0.03)
    p.add_argument("--demo-games", type=int, default=50, help="партий учителя для DQfD")
    p.add_argument("--demo-size", type=int, default=30_000)
    p.add_argument("--demo-ratio", type=float, default=0.5)
    p.add_argument("--aux-coef", type=float, default=0.15)
    p.add_argument("--expert-margin", type=float, default=1.0)
    p.add_argument("--expert-prob", type=float, default=0.8,
                   help="доля ходов учителя в начале обучения")
    p.add_argument("--expert-prob-end", type=float, default=0.25,
                   help="доля ходов учителя в конце обучения")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="cpu / cuda / mps / auto")
    p.add_argument("--load", type=Path, default=None, help="продолжить обучение с чекпоинта")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
