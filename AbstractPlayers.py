from abc import abstractmethod
from enum import Enum

# Probability to generate 4 between {2,4}
PROBABILITY = 0.1


class Turn(Enum):
    MOVE_PLAYER_TURN = 'MOVE_PLAYER'
    INDEX_PLAYER_TURN = 'INDEX_PLAYER'


class Move(Enum):
    LEFT = 'LEFT'
    RIGHT = 'RIGHT'
    UP = 'UP'
    DOWN = 'DOWN'


class AbstractMovePlayer:

    @abstractmethod
    def get_move(self, board, time_limit) -> Move:
        pass


class AbstractIndexPlayer:

    @abstractmethod
    def get_indices(self, board, value, time_limit) -> (int, int):
        pass
