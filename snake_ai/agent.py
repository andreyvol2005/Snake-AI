"""DQN-агент: нейросеть выбирает действие (KEEP / TURN_L / TURN_R)."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .game import ACTIONS, KEEP, TURN_L, TURN_R


@dataclass
class AgentConfig:
    state_kind: str = "vector"
    hidden: int = 256
    lr: float = 1e-3
    gamma: float = 0.95
    batch_size: int = 128
    buffer_size: int = 50_000
    target_sync: int = 1_000      # как часто копировать веса в целевую сеть
    learn_start: int = 1_000      # сколько переходов набрать до начала обучения
    epsilon_start: float = 1.0
    epsilon_min: float = 0.02
    epsilon_decay: float = 0.9995 # множитель на каждый эпизод
    seed: int = 0
    device: str = "auto"


def _pick_device(pref: str) -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class QNetwork(nn.Module):
    """Небольшой MLP: состояние -> оценки трёх действий."""

    def __init__(self, in_features: int, hidden: int, n_actions: int = len(ACTIONS)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    def __init__(self, state_size: int, cfg: Optional[AgentConfig] = None):
        self.cfg = cfg or AgentConfig()
        self.device = _pick_device(self.cfg.device)
        self.online = QNetwork(state_size, self.cfg.hidden).to(self.device)
        self.target = QNetwork(state_size, self.cfg.hidden).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.cfg.lr)
        self.buffer: Deque[Tuple] = deque(maxlen=self.cfg.buffer_size)
        self.epsilon = self.cfg.epsilon_start
        self.steps_done = 0
        self.loss: Optional[float] = None
        if self.cfg.seed is not None:
            torch.manual_seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)
            random.seed(self.cfg.seed)

    # ------------------------------------------------------------- инференс
    @torch.no_grad()
    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        """Выбрать действие. greedy=True — без случайных поворотов (для проверки)."""
        if not greedy and random.random() < self.epsilon:
            return random.choice(ACTIONS)
        x = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return int(self.online(x).argmax(dim=1).item())

    # -------------------------------------------------------------- обучение
    def remember(self, state, action, reward, next_state, done) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay)

    def train_step(self) -> Optional[float]:
        if len(self.buffer) < self.cfg.learn_start:
            return None

        batch = random.sample(self.buffer, self.cfg.batch_size)
        states = torch.as_tensor(np.array([b[0] for b in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([b[1] for b in batch], dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor([b[2] for b in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(np.array([b[3] for b in batch]), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([b[4] for b in batch], dtype=torch.float32, device=self.device).unsqueeze(1)

        # Double DQN: действие выбирает онлайн-сеть, оценку даёт целевая.
        with torch.no_grad():
            next_actions = self.online(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target(next_states).gather(1, next_actions)
            targets = rewards + self.cfg.gamma * next_q * (1.0 - dones)

        q_values = self.online(states).gather(1, actions)
        loss = nn.functional.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.cfg.target_sync == 0:
            self.target.load_state_dict(self.online.state_dict())

        self.loss = float(loss.item())
        return self.loss

    # ------------------------------------------------------ сохранение/загрузка
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.online.state_dict(),
                "cfg": self.cfg.__dict__,
                "epsilon": self.epsilon,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, state_size: int) -> "DQNAgent":
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
        cfg = AgentConfig(**ckpt["cfg"])
        agent = cls(state_size, cfg)
        agent.online.load_state_dict(ckpt["state_dict"])
        agent.target.load_state_dict(ckpt["state_dict"])
        agent.epsilon = ckpt.get("epsilon", 0.0)
        return agent