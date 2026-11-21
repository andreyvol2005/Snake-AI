"""DQN-агент: нейросеть выбирает действие (KEEP / TURN_L / TURN_R).

Для векторного состояния используется MLP, для карты поля — небольшая
свёрточная сеть. У обеих ветвей по три скрытых полносвязных слоя, поэтому
агент может планировать длинные обходы собственного тела.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .game import ACTIONS, KEEP, TURN_L, TURN_R


@dataclass
class AgentConfig:
    state_kind: str = "vector"
    arch: str = "auto"          # auto / mlp / cnn
    hidden: int = 128           # нейронов в скрытых полносвязных слоях
    hidden_layers: int = 3      # сколько скрытых слоёв (не считая выходной)
    lr: float = 3e-4
    gamma: float = 0.99
    batch_size: int = 256
    buffer_size: int = 40_000
    target_sync: int = 2_000    # реже синхронизация целевой сети — стабильнее
    learn_start: int = 5_000    # набрать опыт перед первым обучением
    epsilon_start: float = 1.0
    epsilon_min: float = 0.03
    epsilon_decay: float = 0.9997
    # Демонстрации учителя (DQfD): доля демо-переходов в батче и сила
    # margin-лосса, который держит ход учителя выше остальных.
    demo_ratio: float = 0.5
    aux_coef: float = 0.15
    expert_margin: float = 1.0
    seed: int = 0
    device: str = "auto"
    # Каналы свёрточной ветви: 16 -> 32 -> 32 фильтров.
    conv_channels: Tuple[int, int, int] = (16, 32, 32)


def _pick_device(pref: str) -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class QNetwork(nn.Module):
    """MLP: состояние -> оценки действий. Скрытые активации доступны для панели."""

    def __init__(self, in_features: int, hidden: int, n_actions: int = len(ACTIONS),
                 hidden_layers: int = 3):
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers должен быть >= 1")
        self.in_features = in_features
        self.hidden = hidden
        self.hidden_layers = hidden_layers
        dims = [in_features] + [hidden] * hidden_layers + [n_actions]
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        )

    def _forward_hidden(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        acts: List[torch.Tensor] = []
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
            acts.append(x)
        return self.layers[-1](x), acts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)

    def forward_with_activations(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        return self._forward_hidden(x)


class ConvQNetwork(nn.Module):
    """Свёрточная сеть для карты поля + три полносвязных скрытых слоя.

    Свёртки видят всю доску и учатся узнавать тело, голову, яблоко и
    свободные коридоры независимо от того, где они находятся.
    """

    def __init__(self, in_channels: int, grid: int, hidden: int,
                 n_actions: int = len(ACTIONS), hidden_layers: int = 3,
                 conv_channels: Sequence[int] = (16, 32, 32)):
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers должен быть >= 1")
        c1, c2, c3 = conv_channels
        self.in_channels = in_channels
        self.grid = grid
        self.hidden = hidden
        self.hidden_layers = hidden_layers
        self.conv1 = nn.Conv2d(in_channels, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(c2, c3, 3, stride=2, padding=1)

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, grid, grid)
            flat = int(self._conv_forward(dummy).numel())

        dims = [flat] + [hidden] * hidden_layers + [n_actions]
        self.fcs = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        )

    def _conv_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.flatten(1)

    def _forward_hidden(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        conv_acts: List[torch.Tensor] = []
        x = F.relu(self.conv1(x))
        conv_acts.append(x)
        x = F.relu(self.conv2(x))
        conv_acts.append(x)
        x = F.relu(self.conv3(x))
        conv_acts.append(x)
        x = x.flatten(1)
        fc_acts: List[torch.Tensor] = []
        for layer in self.fcs[:-1]:
            x = F.relu(layer(x))
            fc_acts.append(x)
        return self.fcs[-1](x), conv_acts, fc_acts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv_forward(x)
        for layer in self.fcs[:-1]:
            x = F.relu(layer(x))
        return self.fcs[-1](x)

    def forward_with_activations(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        return self._forward_hidden(x)


class ReplayBuffer:
    """Кольцевой буфер в памяти: быстрый сэмплинг батча без списков Python.

    Массивы фиксированного размера позволяют за один раз копировать сразу
    весь батч, а не собирать его из сотен отдельных массивов. Карты поля
    храним в float16 — это вдвое меньше памяти и быстрее копирование.
    """

    def __init__(self, capacity: int, state_shape: Sequence[int],
                 dtype: np.dtype = np.float16):
        self.capacity = max(1, int(capacity))
        self.state_shape = tuple(int(v) for v in state_shape)
        self.dtype = np.dtype(dtype)
        self.states = np.empty((self.capacity, *self.state_shape), dtype=self.dtype)
        self.next_states = np.empty((self.capacity, *self.state_shape), dtype=self.dtype)
        self.actions = np.empty(self.capacity, dtype=np.int64)
        self.rewards = np.empty(self.capacity, dtype=np.float32)
        self.dones = np.empty(self.capacity, dtype=np.float32)
        self.size = 0
        self.pos = 0

    def __len__(self) -> int:
        return self.size

    def add(self, state, action, reward, next_state, done) -> None:
        i = self.pos
        self.states[i] = state
        self.next_states[i] = next_state
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = 1.0 if done else 0.0
        self.pos = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        idx = np.random.randint(0, self.size, size=min(batch_size, self.size))
        states = torch.from_numpy(self.states[idx]).float().to(device, non_blocking=True)
        next_states = torch.from_numpy(self.next_states[idx]).float().to(device, non_blocking=True)
        actions = torch.from_numpy(self.actions[idx]).to(device, non_blocking=True).unsqueeze(1)
        rewards = torch.from_numpy(self.rewards[idx]).to(device, non_blocking=True).unsqueeze(1)
        dones = torch.from_numpy(self.dones[idx]).to(device, non_blocking=True).unsqueeze(1)
        return states, actions, rewards, next_states, dones


@dataclass
class NetworkSnapshot:
    """Всё, что нужно панели визуализации для одного кадра."""

    arch: str
    chosen_action: int
    output_values: np.ndarray
    # Векторный вход (MLP) или карта поля (CNN).
    input_values: np.ndarray
    # True — действие разрешено щитом безопасности; False — отсечено.
    action_mask: np.ndarray = field(default_factory=lambda: np.ones(len(ACTIONS), dtype=bool))
    head_map: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.float32))
    hidden_activations: List[np.ndarray] = field(default_factory=list)
    feature_values: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    dense_weights: List[np.ndarray] = field(default_factory=list)
    conv_activations: List[np.ndarray] = field(default_factory=list)
    conv_weights: List[np.ndarray] = field(default_factory=list)


class DQNAgent:
    def __init__(self, state_shape: Sequence[int] | int, cfg: Optional[AgentConfig] = None):
        self.cfg = cfg or AgentConfig()
        if isinstance(state_shape, int):
            self.state_shape = (state_shape,)
        else:
            self.state_shape = tuple(int(v) for v in state_shape)
        if not self.state_shape:
            raise ValueError("Пустая форма состояния")

        if self.cfg.arch == "auto":
            self.cfg.arch = "cnn" if len(self.state_shape) == 3 else "mlp"
        if self.cfg.arch == "cnn" and len(self.state_shape) != 3:
            raise ValueError("Для CNN нужно состояние (каналы, высота, ширина)")
        if self.cfg.arch == "mlp" and len(self.state_shape) > 2:
            raise ValueError("MLP ждёт плоское состояние")

        self.device = _pick_device(self.cfg.device)
        self.online = self._build().to(self.device)
        self.target = self._build().to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.cfg.lr)
        self.buffer = ReplayBuffer(self.cfg.buffer_size, self.state_shape)
        self.demo_buffer: Optional[ReplayBuffer] = None
        self.epsilon = self.cfg.epsilon_start
        self.steps_done = 0
        self.loss: Optional[float] = None

        # Для сбора опыта держим копию сети на CPU: на MPS один инференс
        # маленькой сети в 10 раз дороже из-за накладных расходов, а батч
        # для обучения всё равно считается на GPU. Копию обновляем редко.
        self.actor: Optional[nn.Module] = None
        if self.device.type != "cpu":
            self.actor = self._build().to("cpu")
            self.actor.load_state_dict(self.online.state_dict())
            self.actor.eval()
        if self.cfg.seed is not None:
            torch.manual_seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)
            random.seed(self.cfg.seed)

    def _build(self) -> nn.Module:
        if self.cfg.arch == "cnn":
            channels, height, width = self.state_shape
            if height != width:
                raise ValueError("Свёрточная сеть ждёт квадратную карту")
            return ConvQNetwork(
                channels, height, self.cfg.hidden,
                hidden_layers=self.cfg.hidden_layers,
                conv_channels=self.cfg.conv_channels,
            )
        in_features = int(np.prod(self.state_shape))
        return QNetwork(
            in_features, self.cfg.hidden,
            hidden_layers=self.cfg.hidden_layers,
        )

    # ------------------------------------------------------------- инференс
    def _state_tensor(self, state: np.ndarray) -> torch.Tensor:
        x = torch.as_tensor(np.asarray(state), dtype=torch.float32, device=self.device)
        return x.unsqueeze(0)

    def sync_actor(self) -> None:
        """Обновить CPU-копию сети, по которой собирается опыт."""
        if self.actor is not None:
            self.actor.load_state_dict(self.online.state_dict())
            self.actor.eval()

    def _inference_net(self) -> nn.Module:
        return self.actor if self.actor is not None else self.online

    @torch.no_grad()
    def q_values(self, state: np.ndarray) -> np.ndarray:
        """Оценки трёх действий без выбора — нужны щиту безопасности."""
        net = self._inference_net()
        device = next(net.parameters()).device
        x = torch.as_tensor(np.asarray(state), dtype=torch.float32, device=device).unsqueeze(0)
        return net(x)[0].detach().cpu().numpy()

    @torch.no_grad()
    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        """Выбрать действие. greedy=True — без случайных поворотов."""
        if not greedy and random.random() < self.epsilon:
            return random.choice(ACTIONS)
        return int(self.q_values(state).argmax())

    @torch.no_grad()
    def snapshot(self, state: np.ndarray, chosen_action: int,
                 action_mask: Optional[np.ndarray] = None) -> NetworkSnapshot:
        """Посчитать активации и веса для панели визуализации."""
        net = self._inference_net()
        device = next(net.parameters()).device
        x = torch.as_tensor(np.asarray(state), dtype=torch.float32, device=device).unsqueeze(0)
        if action_mask is None:
            mask = np.ones(len(ACTIONS), dtype=bool)
        else:
            mask = np.asarray(action_mask, dtype=bool).copy()
        if self.cfg.arch == "cnn":
            q, conv_acts, fc_acts = net.forward_with_activations(x)
            flat = conv_acts[-1].flatten(1)[0]
            return NetworkSnapshot(
                arch="cnn",
                chosen_action=chosen_action,
                output_values=q[0].detach().cpu().numpy(),
                input_values=np.asarray(state)[0].copy(),  # карта 0/1/-1
                action_mask=mask,
                head_map=np.asarray(state)[1].copy(),
                hidden_activations=[a[0].detach().cpu().numpy() for a in fc_acts],
                feature_values=flat.detach().cpu().numpy(),
                dense_weights=[fc.weight.detach().cpu().numpy() for fc in net.fcs],
                conv_activations=[a[0].detach().cpu().numpy() for a in conv_acts],
                conv_weights=[
                    net.conv1.weight.detach().cpu().numpy(),
                    net.conv2.weight.detach().cpu().numpy(),
                    net.conv3.weight.detach().cpu().numpy(),
                ],
            )

        q, acts = net.forward_with_activations(x)
        return NetworkSnapshot(
            arch="mlp",
            chosen_action=chosen_action,
            output_values=q[0].detach().cpu().numpy(),
            input_values=np.asarray(state).ravel().copy(),
            action_mask=mask,
            hidden_activations=[a[0].detach().cpu().numpy() for a in acts],
            dense_weights=[layer.weight.detach().cpu().numpy() for layer in net.layers],
        )

    # -------------------------------------------------------------- обучение
    def remember(self, state, action, reward, next_state, done) -> None:
        self.buffer.add(state, action, reward, next_state, done)

    def remember_demo(self, state, action, reward, next_state, done) -> None:
        if self.demo_buffer is not None:
            self.demo_buffer.add(state, action, reward, next_state, done)

    def set_demo_buffer(self, demo: ReplayBuffer) -> None:
        self.demo_buffer = demo

    def set_buffer_size(self, capacity: int) -> None:
        """Пересоздать буфер другого размера, сохранив последние переходы."""
        if capacity == self.buffer.capacity:
            return
        old = self.buffer
        new = ReplayBuffer(capacity, self.state_shape)
        keep = min(old.size, capacity)
        if keep:
            # Берём самые свежие переходы в хронологическом порядке.
            order = [(old.pos - keep + i) % old.capacity for i in range(keep)]
            for i in order:
                new.add(old.states[i], old.actions[i], old.rewards[i],
                        old.next_states[i], bool(old.dones[i]))
        self.buffer = new
        self.cfg.buffer_size = capacity

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay)

    def train_step(self) -> Optional[float]:
        has_demo = self.demo_buffer is not None and len(self.demo_buffer) > 0
        if len(self.buffer) < self.cfg.learn_start and not has_demo:
            return None

        demo_count = 0
        batch = None
        if has_demo and self.cfg.demo_ratio > 0:
            demo_count = min(
                len(self.demo_buffer),
                max(1, int(self.cfg.batch_size * self.cfg.demo_ratio)),
            )
        online_count = max(1, self.cfg.batch_size - demo_count)
        if len(self.buffer) > 0:
            batch = self.buffer.sample(online_count, self.device)
        if demo_count:
            demo_batch = self.demo_buffer.sample(demo_count, self.device)
            if batch is None:
                batch = demo_batch
            else:
                batch = tuple(torch.cat([a, b], dim=0) for a, b in zip(batch, demo_batch))
        if batch is None:
            return None

        states, actions, rewards, next_states, dones = batch

        # Double DQN: действие выбирает онлайн-сеть, оценку даёт целевая.
        with torch.no_grad():
            next_q_online = self.online(next_states)
            next_actions = next_q_online.argmax(dim=1, keepdim=True)
            next_q_target = self.target(next_states).gather(1, next_actions)
            targets = rewards + self.cfg.gamma * next_q_target * (1.0 - dones)

        q_all = self.online(states)
        q_values = q_all.gather(1, actions)
        loss = F.smooth_l1_loss(q_values, targets)

        # DQfD: на демонстрациях дополнительно требуем, чтобы ход учителя
        # был лучше любого другого хотя бы на expert_margin.
        if demo_count and self.cfg.aux_coef > 0:
            q_demo = q_all[-demo_count:]
            expert_actions = actions[-demo_count:]
            q_expert = q_demo.gather(1, expert_actions)
            mask = torch.zeros_like(q_demo, dtype=torch.bool)
            mask.scatter_(1, expert_actions, True)
            q_other = q_demo.masked_fill(mask, -1e9).max(dim=1, keepdim=True).values
            margin = F.relu(q_other - q_expert + self.cfg.expert_margin).mean()
            loss = loss + self.cfg.aux_coef * margin

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.cfg.target_sync == 0:
            self.target.load_state_dict(self.online.state_dict())

        # .item() синхронизирует MPS и стоит несколько миллисекунд, поэтому
        # снимаем loss с графа редко — только для логов.
        if self.steps_done % 50 == 0:
            self.loss = float(loss.detach().cpu())
            if self.actor is not None:
                self.sync_actor()
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
                "state_shape": self.state_shape,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, state_shape: Sequence[int] | int) -> "DQNAgent":
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"]
        fields = set(AgentConfig.__dataclass_fields__)
        cfg = AgentConfig(**{k: v for k, v in ckpt.get("cfg", {}).items() if k in fields})
        shape = tuple(ckpt.get("state_shape") or (state_shape if not isinstance(state_shape, int) else (state_shape,)))

        # Определяем архитектуру и глубину по содержимому чекпоинта — так
        # старые модели с двумя скрытыми слоями тоже загружаются.
        if any(k.startswith("conv1.") for k in sd):
            cfg.arch = "cnn"
            fc_ids = sorted({int(m.group(1)) for k in sd
                             if (m := re.match(r"fcs\.(\d+)\.weight", k))})
            cfg.hidden_layers = max(1, len(fc_ids) - 1)
            first = sd["fcs.0.weight"]
            cfg.hidden = int(first.shape[0])
            cfg.conv_channels = (
                int(sd["conv1.weight"].shape[0]),
                int(sd["conv2.weight"].shape[0]),
                int(sd["conv3.weight"].shape[0]),
            )
        else:
            cfg.arch = "mlp"
            old_ids = sorted({int(m.group(1)) for k in sd
                              if (m := re.match(r"fc(\d+)\.weight", k))})
            new_ids = sorted({int(m.group(1)) for k in sd
                              if (m := re.match(r"layers\.(\d+)\.weight", k))})
            count = len(old_ids) or len(new_ids)
            cfg.hidden_layers = max(1, count - 1)
            if old_ids:
                first_key = f"fc1.weight"
            else:
                first_key = "layers.0.weight"
            cfg.hidden = int(sd[first_key].shape[0])

        agent = cls(shape, cfg)
        # Старые чекпоинты хранят слои как fc1/fc2/..., новые — как layers.N.
        mapped = {}
        for key, value in sd.items():
            m = re.match(r"fc(\d+)\.(weight|bias)$", key)
            if m:
                mapped[f"layers.{int(m.group(1)) - 1}.{m.group(2)}"] = value
            else:
                mapped[key] = value
        agent.online.load_state_dict(mapped)
        agent.target.load_state_dict(mapped)
        agent.sync_actor()
        agent.epsilon = ckpt.get("epsilon", 0.0)
        return agent
