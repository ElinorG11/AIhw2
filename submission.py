import logic
import random
from AbstractPlayers import *
import time

# commands to use for move players. dictionary : Move(enum) -> function(board),
# all the functions {up,down,left,right) receive board as parameter and return tuple of (new_board, done, score).
# new_board is according to the step taken, done is true if the step is legal, score is the sum of all numbers that
# combined in this step.
# (you can see GreedyMovePlayer implementation for example)
commands = {Move.UP: logic.up, Move.DOWN: logic.down,
            Move.LEFT: logic.left, Move.RIGHT: logic.right}


# generate value between {2,4} with probability p for 4
def gen_value(p=PROBABILITY):
    return logic.gen_two_or_four(p)


class GreedyMovePlayer(AbstractMovePlayer):
    """Greedy move player provided to you (no need to change),
    the player receives time limit for a single step and the board as parameter and return the next move that gives
    the best score by looking one step ahead.
    """

    def get_move(self, board, time_limit) -> Move:
        optional_moves_score = {}
        for move in Move:
            new_board, done, score = commands[move](board)
            if done:
                optional_moves_score[move] = score

        return max(optional_moves_score, key=optional_moves_score.get)


class RandomIndexPlayer(AbstractIndexPlayer):
    """Random index player provided to you (no need to change),
    the player receives time limit for a single step and the board as parameter and return the next indices to
    put 2 randomly.
    """

    def get_indices(self, board, value, time_limit) -> (int, int):
        a = random.randint(0, len(board) - 1)
        b = random.randint(0, len(board) - 1)
        while board[a][b] != 0:
            a = random.randint(0, len(board) - 1)
            b = random.randint(0, len(board) - 1)
        return a, b


# part A
class ImprovedGreedyMovePlayer(AbstractMovePlayer):
    """Improved greedy Move Player,
    implement get_move function with greedy move that looks only one step ahead with heuristic.
    (you can add helper functions as you want).
    """

    def __init__(self):
        AbstractMovePlayer.__init__(self)

    def calc_bonus(self, params):
        bonus = 0
        for i in range(0, len(params), 2):
            bonus += params[i] * params[i + 1]
        return bonus

    """ Calculates how many empty cells are in the grid """

    def get_empty_slots(self, board):
        return len([item for item in board if item != 0])

    def calc_log(self, num):
        log = 0
        while num > 1:
            num = num // 2
            log = log + 1
        return log

    """ 
    Calculates how much the grid is uniforaml - if all the cells
    are equal, the function will return 0. Otherwise, it will return 
    the inverse of sum of all differences (in absolute value) between neighbours in the grid.
    """

    def get_smoothness(self, board):
        smoothness = 0
        for row in range(len(board)):
            for col in range(len(board)):
                if board[row][col] is not 0:
                    if row + 1 < len(board):
                        smoothness += abs(self.calc_log(board[row][col]) - self.calc_log(board[row + 1][col]))
                    if col + 1 < len(board):
                        smoothness += abs(self.calc_log(board[row][col]) - self.calc_log(board[row][col + 1]))

        return -smoothness

    """ 
    Bonus for tiles in monotonic structure where highest value is in one of the corners.
    """

    def get_monotonicity(self, board):
        monotonicity = 0
        last_row = 3
        last_col = 3
        for row in range(len(board) - 1):
            if board[row][last_col] <= board[row + 1][last_col]:
                monotonicity = monotonicity + 1
            else:
                monotonicity = monotonicity - 20

        for col in range(len(board) - 1):
            if board[last_row][col] <= board[last_row][col + 1]:
                monotonicity = monotonicity + 1
            else:
                monotonicity = monotonicity - 20

        return monotonicity

    def get_monotonicity_weights(self, board):
        weights = [0.165, 0.121, 0.102, 0.0999,
                   0.0997, 0.088, 0.076, 0.0724,
                   0.0606, 0.0562, 0.0371, 0.0161,
                   0.0125, 0.0099, 0.0057, 0.0033]

        cells = []

        for row in range(len(board)):
            for col in range(len(board)):
                cells.append(board[row][col])

        score = []

        for num1, num2 in zip(weights, cells):
            score.append(num1 * num2)

        return sum(score)

    """ 
    Calculates how much the grid is organized in some direction. 
    """

    def direction(self, board):
        grid = [[0 for i in range(len(board))] for j in range(len(board))]
        for row in range(len(board)):
            for col in range(len(board)):
                grid[row][col] = self.calc_log(board[row][col])

        asndud = 0
        dsndud = 0
        asndlr = 0
        dsndlr = 0
        for row in range(4):
            for col in range(4):
                if col + 1 < 4:
                    if grid[row][col] > grid[row][col + 1]:
                        dsndlr -= grid[row][col] - grid[row][col + 1]
                    else:
                        asndlr += grid[row][col] - grid[row][col + 1]
                if row + 1 < 4:
                    if grid[row][col] > grid[row + 1][col]:
                        dsndud -= grid[row][col] - grid[row + 1][col]
                    else:
                        asndud += grid[row][col] - grid[row + 1][col]
        return max(dsndlr, asndlr) + max(dsndud, asndud)

    """ Returns the maximal tile. """

    def highestTile(self, board) -> int:
        return max(max(board))

    def heuristic(self, board):
        monotonicityFact = 0.25
        smoothnessFact = 0.15
        emptyFact = 0.35
        highestFact = 0.1
        directionFact = 0.15

        # calc monotonicity (snake)
        # monotonicity = self.get_monotonicity(board)
        monotonicity = self.get_monotonicity_weights(board)
        # calc smoothness
        smoothness = self.get_smoothness(board)
        # calc empty slots
        empty_slots = self.get_empty_slots(board)
        direction = self.direction(board)

        max_tile = self.highestTile(board)

        params = [monotonicityFact, monotonicity, smoothnessFact, smoothness, emptyFact, empty_slots, directionFact, direction, highestFact, max_tile]

        new_bonus = self.calc_bonus(params)
        print("Monotonicity bonus = " + str(monotonicity) + " Smoothness = " + str(smoothness) + " Empty-slots = " + str(empty_slots) + " Direction = " + str(direction))
        print("Bonus = " + str(new_bonus))
        return new_bonus / 10

    def get_move(self, board, time_limit) -> Move:
        optional_moves_score = {}
        for move in Move:
            new_board, done, score = commands[move](board)
            if done:
                # optional_moves_score[move] = 1.4 * self.heuristic(board) + 1.4 * score
                optional_moves_score[move] = 0.15 * self.heuristic(board) + 0.85 * score

        return max(optional_moves_score, key=optional_moves_score.get)


def minimax_search(state, depth, agent):
    if depth == 0 or MiniMaxMovePlayer.is_goal(board=state) is True:
        minimax_agent = MiniMaxMovePlayer()
        return minimax_agent.heuristic(state)
    if agent == Turn.MOVE_PLAYER_TURN:
        curr_max = float('-inf')
        for move in Move:
            # store previous state of the board
            prev_state = [[state[i][j] for i in range(4)] for j in range(4)]
            # change the board
            new_state, valid, score = commands[move](state)
            if valid:
                value = minimax_search(state, depth - 1, agent)
                state = [[prev_state[i][j] for i in range(4)] for j in range(4)]
                curr_max = max(curr_max, value)
        return curr_max
    else:
        curr_min = float('inf')
        for move in Move:
            # store previous state of the board
            prev_state = [[state[i][j] for i in range(4)] for j in range(4)]
            # change the board
            new_state, valid, score = commands[move](state)
            if valid:
                value = minimax_search(state, depth - 1, agent)
                state = [[prev_state[i][j] for i in range(4)] for j in range(4)]
                curr_min = min(curr_min, value)
        return curr_min


def search(state, depth, agent):
    if agent == Turn.MOVE_PLAYER_TURN:
        max_value, best_move = float('-inf'), None
        for move in Move:
            # store previous state of the board
            prev_state = [[state[i][j] for i in range(4)] for j in range(4)]

            # change the board
            new_state, valid, score = commands[move](state)

            if valid:
                cur_minimax_val = minimax_search(state, depth - 1, agent)
                state = [[prev_state[i][j] for i in range(4)] for j in range(4)]

                if cur_minimax_val >= max_value:
                    max_value = cur_minimax_val
                    best_move = move
        return best_move, max_value
    else:
        min_value, a, b = float('inf'), 0, 0
        for i in range(len(state)):
            for j in range(len(state)):
                # store previous state of the board
                prev_state = [[state[i][j] for i in range(len(state))] for j in range(len(state))]

                # change the board
                if state[i][j] is 0:
                    state[i][j] += 2
                    cur_minimax_val = minimax_search(state, depth - 1, agent)
                    state = [[prev_state[i][j] for i in range(len(state))] for j in range(len(state))]

                    if cur_minimax_val <= min_value:
                        min_value = cur_minimax_val
                        a, b = i, j
        return min_value, a, b


# part B
class MiniMaxMovePlayer(AbstractMovePlayer):
    """MiniMax Move Player,
    implement get_move function according to MiniMax algorithm
    (you can add helper functions as you want).
    """

    def __init__(self):
        AbstractMovePlayer.__init__(self)

    def get_move(self, board, time_limit) -> Move:
        time_start = time.time()
        depth = 1
        max_move, max_val = search(board, depth, Turn.MOVE_PLAYER_TURN)
        last_iteration_time = time.time() - time_start
        next_iteration_max_time = 4 * last_iteration_time
        time_until_now = time.time() - time_start
        while time_until_now + next_iteration_max_time < time_limit:
            depth += 1
            iteration_start_time = time.time()
            last_good_move = max_move
            max_move, val = search(board, depth, Turn.MOVE_PLAYER_TURN)
            if val == float('inf'):
                break
            if val == float('-inf'):
                max_move = last_good_move
                break
            last_iteration_time = time.time() - iteration_start_time
            next_iteration_max_time = 4 * last_iteration_time
            time_until_now = time.time() - time_start
        return max_move

    def calc_bonus(self, params):
        bonus = 0
        for i in range(0, len(params), 2):
            bonus += params[i] * params[i + 1]
        return bonus

    """ Calculates how many empty cells are in the grid """

    def get_empty_slots(self, board):
        empty_slots = [item for item in board if item != 0]
        return 16-len(empty_slots)

    def calc_log(self, num):
        log = 0
        while num > 1:
            num = num / 2
            log = log + 1
        return log

    """ 
    Calculates how much the grid is uniforaml - if all the cells
    are equal, the function will return 0. Otherwise, it will return 
    the inverse of sum of all differences (in abolute value) between neighbours in the grid.
    """

    def get_smoothness(self, board):
        smoothness = 0
        for row in range(len(board)):
            for col in range(len(board)):
                if board[row][col] is not 0:
                    if board[row][3] is not 0:
                        smoothness += abs(self.calc_log(board[row][col]) - self.calc_log(board[row][3]))
                    if board[3][col] is not 0:
                        smoothness += abs(self.calc_log(board[row][col]) - self.calc_log(board[3][col]))

        return -smoothness

    """ 
    Bonus for tiles in monotonic structure where highest value is in one of the corners.
    """

    def get_monotonicity(self, board):
        monotonicity = 0
        last_row = 3
        last_col = 3
        for row in range(len(board)-1):
            if board[row][last_col] <= board[row + 1][last_col]:
                monotonicity = monotonicity + 1
            else:
                monotonicity = monotonicity - 2

        for col in range(len(board)-1):
            if board[last_row][col] <= board[last_row][col + 1]:
                monotonicity = monotonicity + 1
            else:
                monotonicity = monotonicity - 2

        return monotonicity

    """ 
    Calculates how much the grid is organized in some direction. 
    """

    def direction(self, board):
        grid = [[0 for i in range(len(board))] for j in range(len(board))]
        for row in range(len(board)):
            for col in range(len(board)):
                grid[row][col] = self.calc_log(board[row][col])

        asnd_ud = 0
        dsnd_ud = 0
        asnd_lr = 0
        dsnd_lr = 0
        for row in range(4):
            for col in range(4):
                if col + 1 < 4:
                    if grid[row][col] > grid[row][col + 1]:
                        dsnd_lr -= grid[row][col] - grid[row][col + 1]
                    else:
                        asnd_lr += grid[row][col] - grid[row][col + 1]
                if row + 1 < 4:
                    if grid[row][col] > grid[row + 1][col]:
                        dsnd_ud -= grid[row][col] - grid[row + 1][col]
                    else:
                        asnd_ud += grid[row][col] - grid[row + 1][col]
        return max(dsnd_lr, asnd_lr) + max(dsnd_ud, asnd_ud)

    """ Returns the maximal tile. """

    def highestTile(self, board) -> int:
        return max(max(board))

    def heuristic(self, board):
        monotonicityFact = 10
        smoothnessFact = 1
        emptyFact = 25
        highestFact = 10
        directionFact = 15

        # calc monotonicity (snake)
        monotonicity = self.get_monotonicity(board)
        # calc smoothness
        smoothness = self.get_smoothness(board)
        # calc empty slots
        empty_slots = self.get_empty_slots(board)
        direction = self.direction(board)

        max_tile = self.highestTile(board)

        params = [monotonicityFact, monotonicity, smoothnessFact, smoothness, emptyFact, empty_slots, directionFact, direction, highestFact, max_tile]
        #params = [monotonicityFact, monotonicity, smoothnessFact, smoothness, emptyFact, empty_slots, highestFact, max_tile]

        new_bonus = self.calc_bonus(params)
        #print("Monotonicity bonus = " + str(3) + " Smoothness = " + str(smoothness) + " Empty-slots = " + str(empty_slots) + " Direction = " + str(direction))

        return new_bonus

    def is_goal(board) -> bool:
        return logic.game_state(board) == 'lose'


class MiniMaxIndexPlayer(AbstractIndexPlayer):
    """MiniMax Index Player,
    this player is the opponent of the move player and need to return the indices on the board where to put 2.
    the goal of the player is to reduce move player score.
    implement get_indices function according to MiniMax algorithm, the value in minimax player value is only 2.
    (you can add helper functions as you want).
    """

    def __init__(self):
        AbstractIndexPlayer.__init__(self)

    def get_indices(self, board, value, time_limit) -> (int, int):
        time_start = time.time()
        depth = 1
        min_val, a, b = search(board, depth, Turn.INDEX_PLAYER_TURN)
        last_iteration_time = time.time() - time_start
        next_iteration_max_time = 4 * last_iteration_time
        time_until_now = time.time() - time_start
        while time_until_now + next_iteration_max_time < time_limit:
            depth += 1
            iteration_start_time = time.time()
            x, y = a, b
            val, a, b = search(board, depth, Turn.INDEX_PLAYER_TURN)
            if val == float('inf'):
                a, b = x, y
                break
            if val == float('-inf'):
                break
            last_iteration_time = time.time() - iteration_start_time
            next_iteration_max_time = 4 * last_iteration_time
            time_until_now = time.time() - time_start
        return a, b



# part C
class ABMovePlayer(AbstractMovePlayer):
    """Alpha Beta Move Player,
    implement get_move function according to Alpha Beta MiniMax algorithm
    (you can add helper functions as you want)
    """

    def __init__(self):
        AbstractMovePlayer.__init__(self)
        # TODO: add here if needed

    def get_move(self, board, time_limit) -> Move:
        # TODO: erase the following line and implement this function.
        raise NotImplementedError

    # TODO: add here helper functions in class, if needed


# part D
class ExpectimaxMovePlayer(AbstractMovePlayer):
    """Expectimax Move Player,
    implement get_move function according to Expectimax algorithm.
    (you can add helper functions as you want)
    """

    def __init__(self):
        AbstractMovePlayer.__init__(self)
        # TODO: add here if needed

    def get_move(self, board, time_limit) -> Move:
        # TODO: erase the following line and implement this function.
        raise NotImplementedError

    # TODO: add here helper functions in class, if needed


class ExpectimaxIndexPlayer(AbstractIndexPlayer):
    """Expectimax Index Player
    implement get_indices function according to Expectimax algorithm, the value is number between {2,4}.
    (you can add helper functions as you want)
    """

    def __init__(self):
        AbstractIndexPlayer.__init__(self)
        # TODO: add here if needed

    def get_indices(self, board, value, time_limit) -> (int, int):
        # TODO: erase the following line and implement this function.
        raise NotImplementedError

    # TODO: add here helper functions in class, if needed


# Tournament
class ContestMovePlayer(AbstractMovePlayer):
    """Contest Move Player,
    implement get_move function as you want to compete in the Tournament
    (you can add helper functions as you want)
    """

    def __init__(self):
        AbstractMovePlayer.__init__(self)
        # TODO: add here if needed

    def get_move(self, board, time_limit) -> Move:
        # TODO: erase the following line and implement this function.
        raise NotImplementedError

    # TODO: add here helper functions in class, if needed
