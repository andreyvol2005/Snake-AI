"""ИИ для «Змейки»: DQN-агент, который учится играть сам."""

from .agent import AgentConfig, DQNAgent
from .game import ACTIONS, GameConfig, GameView, SnakeGame
from .state import encode, state_size

__all__ = [
    "AgentConfig",
    "DQNAgent",
    "GameConfig",
    "GameView",
    "SnakeGame",
    "ACTIONS",
    "encode",
    "state_size",
]