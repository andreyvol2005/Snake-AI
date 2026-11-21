"""DQN-агент: нейросеть выбирает действие (KEEP / TURN_L / TURN_R)."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .game import ACTIONS, KEEP, TURN_L, TURN_R


@dataclass
class AgentConfig:
    state_kind: str = "vector"
    hidden: int = 128          # меньше нейронов — быстрее сходится и легче визуализировать
    lr: float = 1e-3
    gamma: float = 0.97
    batch_size: int = 256
    buffer_size: int = 100_000
    target_sync: int = 2_000   # реже синхронизация целевой сети — стабильнее
    learn_start: int = 2_000   # набрать больше опыта перед первым обучением
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05  # выше минимум — агент не перестаёт探роваться слишком рано
    epsilon_decay: float = 0.9998  # медленнее затухание — дольше ищет стратегию
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
    """MLP: состояние -> оценки трёх действий. Промежуточные активации доступны для визуализации."""

    def __init__(self, in_features: int, hidden: int, n_actions: int = len(ACTIONS)):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_actions)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

    def forward_with_activations(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Возвращает выход сети и активации после каждого слоя (для визуализации)."""
        acts: List[torch.Tensor] = []
        x = self.fc1(x)
        acts.append(self.relu(x))
        x = self.fc2(acts[-1])
        acts.append(self.relu(x))
        x = self.fc3(acts[-1])
        return x, acts


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
        q, _ = self.online.forward_with_activations(x)
        return int(q.argmax(dim=1).item())

    @torch.no_grad()
    def act_with_activations(
        self, state: np.ndarray, greedy: bool = False
    ) -> Tuple[int, List[np.ndarray]]:
        """То же, что act, но возвращает активации слоёв для визуализации."""
        x = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        if not greedy and random.random() < self.epsilon:
            action = random.choice(ACTIONS)
            q = torch.zeros(1, len(ACTIONS), device=self.device)
        else:
            q, _ = self.online.forward_with_activations(x)
            action = int(q.argmax(dim=1).item())

        # Пересчитываем активации без градиента для отображения.
        _, acts = self.online.forward_with_activations(x)
        activations = [a.detach().cpu().numpy().ravel() for a in acts]
        return action, activations

    def get_weights(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Возвращает (веса, смещения) каждого линейного слоя для визуализации связей."""
        pairs: List[Tuple[np.ndarray, np.ndarray]] = []
        for layer in (self.online.fc1, self.online.fc2, self.online.fc3):
            pairs.append(
                (layer.weight.detach().cpu().numpy(), layer.bias.detach().cpu().numpy())
            )
        return pairs

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
            next_q_online = self.online(next_states)
            next_actions = next_q_online.argmax(dim=1, keepdim=True)
            next_q_target = self.target(next_states).gather(1, next_actions)
            targets = rewards + self.cfg.gamma * next_q_target * (1.0 - dones)

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
        # Отбрасываем лишние ключи, если чекпоинт сохранён другой версией.
        fields = set(AgentConfig.__dataclass_fields__)
        cfg = AgentConfig(**{k: v for k, v in ckpt["cfg"].items() if k in fields})
        agent = cls(state_size, cfg)
        agent.online.load_state_dict(ckpt["state_dict"])
        agent.target.load_state_dict(ckpt["state_dict"])
        agent.epsilon = ckpt.get("epsilon", 0.0)
        return agent