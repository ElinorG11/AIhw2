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
                if board[row][col] != 0:
                    if row + 1 < len(board):
                        smoothness += abs(self.calc_log(board[row][col]) - self.calc_log(board[row + 1][col]))
                    if col + 1 < len(board):
                        smoothness += abs(self.calc_log(board[row][col]) - self.calc_log(board[row][col + 1]))

        return -smoothness

    """ 
    Bonus for tiles in monotonic structure where highest value is in one of the corners.
    """

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
        monotonicity = self.get_monotonicity_weights(board)
        # calc smoothness
        smoothness = self.get_smoothness(board)
        # calc empty slots
        empty_slots = self.get_empty_slots(board)
        direction = self.direction(board)

        max_tile = self.highestTile(board)

        params = [monotonicityFact, monotonicity, smoothnessFact, smoothness, emptyFact, empty_slots, directionFact,
                  direction, highestFact, max_tile]

        new_bonus = self.calc_bonus(params)
        return new_bonus / 10

    def get_move(self, board, time_limit) -> Move:
        optional_moves_score = {}
        for move in Move:
            new_board, done, score = commands[move](board)
            if done:
                optional_moves_score[move] = 0.15 * self.heuristic(board) + 0.85 * score
        return max(optional_moves_score, key=optional_moves_score.get)


# part B
class MiniMaxMovePlayer(AbstractMovePlayer):
    """MiniMax Move Player,
    implement get_move function according to MiniMax algorithm
    (you can add helper functions as you want).
    """

    def __init__(self):
        AbstractMovePlayer.__init__(self)
        # fields used for error debugging & calculating average depth for q. 12 in dry assignment
        self.depth_sums = 0
        self.move_count = 0

    def get_move(self, board, time_limit) -> Move:
        time_start = time.time()
        depth = 1
        max_val, max_move = self.MinimaxSearch((board, Turn.MOVE_PLAYER_TURN), Turn.MOVE_PLAYER_TURN, depth)
        curr_iter_time = time.time() - time_start
        """
        since different agents have different branching factors (upto 4 for player, upto 14-15 for computer)
        then to effectively bound the time we look at the time from the previous calculation for the
        same agent (a.k.a 2 iterations before hand) and multiply that by a maximum branching factor of 
        10*4 = 40. 
        For small depth-numbers, node_ratio may be bigger than this, since there may be 11-15 open spaces and all the
        non-leaf nodes in the tree effect the total number of nodes in the tree more significantly but considering
        the fact that for small depth numbers we won't timeout anyway (since timout>=1 is given) 
        then we allow ourselves to make this rounded down estimate

        to be completely safe we can choose 15*4, but our testing works consistently with 10*4 as well
        """
        node_ratio = 4 * 15
        prev_iter_time = curr_iter_time
        while node_ratio * prev_iter_time < time_limit - (time.time() - time_start) - 0.01:
            depth += 1
            #print("(Player) curr depth is: " + str(depth))
            iteration_start_time = time.time()
            last_good_move = max_move
            val, max_move = self.MinimaxSearch((board, Turn.MOVE_PLAYER_TURN), Turn.MOVE_PLAYER_TURN, depth)
            if val == float('inf'):
                break
            if val == float('-inf'):
                max_move = last_good_move
                break
            prev_iter_time = curr_iter_time
            # using max here to make sure curr_iter_time always rises, to avoid 'noise'
            # where the current iteration as faster than the previous one even though
            # it has a greater tree-depth to search
            curr_iter_time = time.time() - iteration_start_time
            #next_iteration_max_time = node_ratio * prev_iter_time
            #print("prev iter time: " + str(prev_iter_time) + " curr iter time: " + str(
            #    curr_iter_time) + " next max time: " + str(next_iteration_max_time))
        #self.move_count += 1
        #self.depth_sums += depth
        return max_move

    def MinimaxSearch(self, state, agent, depth):
        board = state[0]
        agentToMove = state[1]
        if self.is_goal(state):
            return float('-inf'), None
        if depth == 0:
            return self.heuristic(state[0]), None
        turn = agentToMove
        best_move = None
        if turn == agent:
            curr_max = float('-inf')
            for move in Move:
                new_board, valid, score = commands[move](self.copy_board(board))
                if valid:
                    new_state = (new_board, Turn.INDEX_PLAYER_TURN)
                    val, _ = self.MinimaxSearch(new_state, agent, depth - 1)
                    if val >= curr_max:
                        curr_max = val
                        best_move = move
            return curr_max, best_move
        else:
            cur_min = float("inf")
            for (i, j) in self.get_empty_indices(board):
                new_board = self.copy_board(board)
                new_board[i][j] = 2
                new_state = (new_board, Turn.MOVE_PLAYER_TURN)
                val, _ = self.MinimaxSearch(new_state, agent, depth - 1)
                if val <= cur_min:
                    cur_min = val
                    best_move = (i, j)
            return cur_min, best_move

    def get_empty_indices(self, board):
        empty = []
        for i in range(0, 4):
            for j in range(0, 4):
                if board[i][j] == 0:
                    empty.append((i, j))
        return empty

    def heuristic(self, board):
        filled_score_map = [[1, 0.5, 0.5, 1],
                            [0.5, 0.2, 0.2, 0.5],
                            [0.5, 0.2, 0.2, 0.5],
                            [1, 0.5, 0.5, 1]]
        empty_score_map = [[0, 0.5, 0.5, 0],
                           [0.5, 0.8, 0.8, 0.5],
                           [0.5, 0.8, 0.8, 0.5],
                           [0, 0.5, 0.5, 0]]
        filled_score = 0
        empty_score = 0
        for row in range(len(board)):
            for col in range(len(board)):
                filled_score += (board[row][col] ** 2) * filled_score_map[row][col]
                if board[row][col] == 0:
                    empty_score = empty_score + 256 * empty_score_map[row][col]

        return 0.45 * filled_score + 0.55 * empty_score

    def is_goal(self, state) -> bool:
        return logic.game_state(state[0]) == 'lose'

    def copy_board(self, board):
        new_board = []
        for i in range(4):
            new_board.append([0] * 4)
        for i in range(4):
            for j in range(4):
                new_board[i][j] = board[i][j]
        return new_board


class MiniMaxIndexPlayer(AbstractIndexPlayer):
    """MiniMax Index Player,
    this player is the opponent of the move player and need to return the indices on the board where to put 2.
    the goal of the player is to reduce move player score.
    implement get_indices function according to MiniMax algorithm, the value in minimax player value is only 2.
    (you can add helper functions as you want).
    """

    def __init__(self):
        AbstractIndexPlayer.__init__(self)
        # fields used for error debugging & calculating average depth for q. 12 in dry assignment
        self.depth_sums = 0
        self.move_count = 0

    def get_indices(self, board, value, time_limit) -> (int, int):
        time_start = time.time()
        depth = 1
        min_val, min_move = self.MinimaxSearch((board, Turn.INDEX_PLAYER_TURN), Turn.INDEX_PLAYER_TURN, depth)
        curr_iter_time = time.time() - time_start
        node_ratio = 4 * 15
        prev_iter_time = curr_iter_time
        while node_ratio * prev_iter_time < time_limit - (time.time() - time_start) - 0.01:
            depth += 1
            #print("(Comp) curr depth is: " + str(depth))
            iteration_start_time = time.time()
            last_good_indices = min_move
            val, min_move = self.MinimaxSearch((board, Turn.INDEX_PLAYER_TURN), Turn.INDEX_PLAYER_TURN, depth)
            if val == float('inf'):  # found winning moves (only possible for index)
                break
            if val == float('-inf'):  # not possible for index
                min_move = last_good_indices
                break
            prev_iter_time = curr_iter_time
            curr_iter_time = time.time() - iteration_start_time
            next_iteration_max_time = node_ratio * prev_iter_time
            #print("prev iter time: " + str(prev_iter_time) + " curr iter time: " + str(
            #    curr_iter_time) + " next max time: " + str(next_iteration_max_time))
        #self.move_count += 1
        #self.depth_sums += depth
        return min_move

    def MinimaxSearch(self, state, agent, depth):
        board = state[0]
        agentToMove = state[1]
        if self.is_goal(state):  # can only reach goal as a result of index's move, so we return inf to reinforce this
            # move (goal state is good for index)
            return float('inf'), None

        if depth == 0:
            return self.heuristic(state[0]), None
        turn = agentToMove
        best_move = None
        if turn == agent:
            curr_max = float('-inf')
            for (i, j) in self.get_empty_indices(board):
                new_board = (self.copy_board(board))
                new_board[i][j] = 2
                new_state = (new_board, Turn.MOVE_PLAYER_TURN)
                val, _ = self.MinimaxSearch(new_state, agent, depth - 1)
                if val >= curr_max:
                    curr_max = val
                    best_move = (i, j)
            return curr_max, best_move
        else:
            cur_min = float('inf')
            for move in Move:
                new_board, valid, score = commands[move](self.copy_board(board))
                if valid:
                    new_state = (new_board, Turn.INDEX_PLAYER_TURN)
                    val, _ = self.MinimaxSearch(new_state, agent, depth - 1)
                    if val <= cur_min:
                        cur_min = val
                        best_move = move
            return cur_min, best_move

    def get_empty_indices(self, board):
        empty = []
        for i in range(0, 4):
            for j in range(0, 4):
                if board[i][j] == 0:
                    empty.append((i, j))
        return empty

    def is_goal(self, state) -> bool:
        return logic.game_state(state[0]) == 'lose'

    def heuristic(self, board):
        filled_score_map = [[1, 0.5, 0.5, 1],
                            [0.5, 0.2, 0.2, 0.5],
                            [0.5, 0.2, 0.2, 0.5],
                            [1, 0.5, 0.5, 1]]
        empty_score_map = [[0, 0.5, 0.5, 0],
                           [0.5, 0.8, 0.8, 0.5],
                           [0.5, 0.8, 0.8, 0.5],
                           [0, 0.5, 0.5, 0]]
        filled_score = 0
        empty_score = 0
        for row in range(len(board)):
            for col in range(len(board)):
                filled_score += (board[row][col] ** 2) * filled_score_map[row][col]
                if board[row][col] == 0:
                    empty_score = empty_score + 256 * empty_score_map[row][col]

        return -1 * (0.45 * filled_score + 0.55 * empty_score)

    def copy_board(self, board):
        new_board = []
        for i in range(4):
            new_board.append([0] * 4)
        for i in range(4):
            for j in range(4):
                new_board[i][j] = board[i][j]
        return new_board


# part C
class ABMovePlayer(AbstractMovePlayer):
    """Alpha Beta Move Player,
    implement get_move function according to Alpha Beta MiniMax algorithm
    (you can add helper functions as you want)
    """

    def __init__(self):
        AbstractMovePlayer.__init__(self)
        # fields used for error debugging & calculating average depth for q. 12 in dry assignment
        #self.depth_sums = 0
        #self.move_count = 0

    def get_move(self, board, time_limit) -> Move:
        time_start = time.time()
        depth = 1
        val, max_move = self.ABminimaxsearch((board, Turn.MOVE_PLAYER_TURN), Turn.MOVE_PLAYER_TURN, depth,
                                             float("-inf"), float("inf"))
        # baseline time that we can use to estimate the next depth time
        curr_iter_time = time.time() - time_start
        node_ratio = 4 * 15
        prev_iter_time = curr_iter_time
        while node_ratio * prev_iter_time < time_limit - (time.time() - time_start) - 0.01:
            depth += 1
            #print("(Player) curr depth is: " + str(depth))
            iteration_start_time = time.time()
            last_good_move = max_move
            val, max_move = self.ABminimaxsearch((board, Turn.MOVE_PLAYER_TURN), Turn.MOVE_PLAYER_TURN, depth,
                                                 float("-inf"), float("inf"))
            if val == float('inf'):
                break
            if val == float('-inf'):
                max_move = last_good_move
                break
            prev_iter_time = curr_iter_time
            curr_iter_time = time.time() - iteration_start_time
            next_iteration_max_time = node_ratio * prev_iter_time
            #print("prev iter time: " + str(prev_iter_time) + " curr iter time: " + str(
            #    curr_iter_time) + " next max time: " + str(next_iteration_max_time))
        #self.move_count += 1
        #self.depth_sums += depth
        return max_move

    # TODO: add here helper functions in class, if needed
    def ABminimaxsearch(self, state, agent, depth, alpha, beta):
        board = state[0]
        curr_turn = state[1]

        if self.is_goal(state):
            return float('-inf'), None
        if depth == 0:
            return self.heuristic(board), None

        turn = curr_turn
        best_move = None
        if turn == agent:
            cur_max = float("-inf")
            for move in Move:
                new_board, valid, score = commands[move](self.copy_board(board))
                if valid:
                    child_state = (new_board, Turn.INDEX_PLAYER_TURN)
                    v, _ = self.ABminimaxsearch(child_state, agent, depth - 1, alpha, beta)
                    if v >= cur_max:
                        cur_max = v
                        best_move = move
                    alpha = max(cur_max, alpha)
                    if cur_max >= beta:
                        return float("inf"), move
            return cur_max, best_move
        else:
            cur_min = float("inf")
            for (i, j, v) in self.get_empty_indexes(board):
                new_board = self.copy_board(board)
                new_board[i][j] = v
                child_state = (new_board, Turn.MOVE_PLAYER_TURN)
                v, _ = self.ABminimaxsearch(child_state, agent, depth - 1, alpha, beta)
                if v <= cur_min:
                    cur_min = v
                    best_move = (i, j, v)
                beta = min(cur_min, beta)
                if cur_min <= alpha:
                    return float("-inf"), (i, j, v)
            return cur_min, best_move

    def heuristic(self, board):
        filled_score_map = [[1, 0.5, 0.5, 1],
                            [0.5, 0.2, 0.2, 0.5],
                            [0.5, 0.2, 0.2, 0.5],
                            [1, 0.5, 0.5, 1]]
        empty_score_map = [[0, 0.5, 0.5, 0],
                           [0.5, 0.8, 0.8, 0.5],
                           [0.5, 0.8, 0.8, 0.5],
                           [0, 0.5, 0.5, 0]]
        filled_score = 0
        empty_score = 0
        for row in range(len(board)):
            for col in range(len(board)):
                filled_score += (board[row][col] ** 2) * filled_score_map[row][col]
                if board[row][col] == 0:
                    empty_score = empty_score + 256 * empty_score_map[row][col]

        return 0.45 * filled_score + 0.55 * empty_score

    def is_goal(self, state) -> bool:
        return logic.game_state(state[0]) == 'lose'

    def get_empty_indexes(self, board):
        empty = []
        for i in range(0, 4):
            for j in range(0, 4):
                if board[i][j] == 0:
                    empty.append((i, j, 2))
                    # empty.append((i, j, 4))
        return empty

    def copy_board(self, board):
        new_board = []
        for i in range(4):
            new_board.append([0] * 4)
        for i in range(4):
            for j in range(4):
                new_board[i][j] = board[i][j]
        return new_board


# part D
class ExpectimaxMovePlayer(AbstractMovePlayer):
    """Expectimax Move Player,
    implement get_move function according to Expectimax algorithm.
    (you can add helper functions as you want)
    """

    def __init__(self):
        AbstractMovePlayer.__init__(self)
        # fields used for error debugging & calculating average depth for q. 12 in dry assignment
        #self.depth_sums = 0
        #self.move_count = 0

    def get_move(self, board, time_limit) -> Move:
        time_start = time.time()
        depth = 1
        max_val, max_move = self.ExpectimaxSearch((board, Turn.MOVE_PLAYER_TURN, 0), Turn.MOVE_PLAYER_TURN, depth)
        curr_iter_time = time.time() - time_start
        """" the timeout logic is the same as before, except that now we add another branching factor of 2 for the
        tile probability choice. now since we hae 3 different branching factor, to estimate the next max iteration
        time, we look at two iterations beforehand and multiply by the total branching factor from then, meaning
        2*4*15"""
        node_ratio = 2 * 4 * 15
        prev_iter_time = curr_iter_time
        prev_prev_iter_time = curr_iter_time
        while node_ratio * prev_prev_iter_time < time_limit - (time.time() - time_start) - 0.01:
            depth += 1
            # print("(Player) curr depth is: " + str(depth))
            iteration_start_time = time.time()
            last_good_move = max_move
            val, max_move = self.ExpectimaxSearch((board, Turn.MOVE_PLAYER_TURN, 0), Turn.MOVE_PLAYER_TURN, depth)
            if val == float('inf'):
                break
            if val == float('-inf'):
                max_move = last_good_move
                break
            prev_prev_iter_time = prev_iter_time
            prev_iter_time = curr_iter_time
            next_iteration_max_time = node_ratio * prev_prev_iter_time
            curr_iter_time = time.time() - iteration_start_time
            # print(" prev prev iter time: " + str(prev_prev_iter_time) + " curr iter time: " + str(
            #    curr_iter_time) + " next max time: " + str(next_iteration_max_time))
        #self.move_count += 1
        #self.depth_sums += depth
        return max_move

    def ExpectimaxSearch(self, state, agent, depth):
        board = state[0]
        agentToMove = state[1]
        info_flag = state[2]  # holds 1 if probability state,  0 in move_player state and 2/4 in index state

        if self.is_goal(state):
            return float('-inf'), None

        if depth == 0:
            return self.heuristic(state[0]), None

        if info_flag == 1:  # probability state appears before the index player's turn
            val2, _ = self.ExpectimaxSearch(((self.copy_board(board)), Turn.INDEX_PLAYER_TURN, 2), agent, depth - 1)
            val4, _ = self.ExpectimaxSearch(((self.copy_board(board)), Turn.INDEX_PLAYER_TURN, 4), agent, depth - 1)
            exp = 0.9 * val2 + 0.1 * val4
            """since the 2/4 choice is not a move and the initial caller doesn't have a turn of type "random choice"
             we can return a None best move"""
            return exp, None
        turn = agentToMove
        best_move = None
        if turn == agent:
            curr_max = float('-inf')
            for move in Move:
                new_board, valid, score = commands[move](self.copy_board(board))
                if valid:
                    new_state = (new_board, Turn.INDEX_PLAYER_TURN, 1)  # next state is a probability state
                    val, _ = self.ExpectimaxSearch(new_state, agent, depth - 1)
                    if val >= curr_max:
                        curr_max = val
                        best_move = move
            return curr_max, best_move
        else:  # index player's turn
            cur_min = float("inf")
            for (i, j) in self.get_empty_indices(board):
                new_board = self.copy_board(board)
                new_board[i][j] = info_flag
                new_state = (new_board, Turn.MOVE_PLAYER_TURN, 0)
                val, _ = self.ExpectimaxSearch(new_state, agent, depth - 1)
                if val <= cur_min:
                    cur_min = val
                    best_move = (i, j)
            return cur_min, best_move

    def get_empty_indices(self, board):
        empty = []
        for i in range(0, 4):
            for j in range(0, 4):
                if board[i][j] == 0:
                    empty.append((i, j))
        return empty

    def heuristic(self, board):
        filled_score_map = [[1, 0.5, 0.5, 1],
                            [0.5, 0.2, 0.2, 0.5],
                            [0.5, 0.2, 0.2, 0.5],
                            [1, 0.5, 0.5, 1]]
        empty_score_map = [[0, 0.5, 0.5, 0],
                           [0.5, 0.8, 0.8, 0.5],
                           [0.5, 0.8, 0.8, 0.5],
                           [0, 0.5, 0.5, 0]]
        filled_score = 0
        empty_score = 0
        for row in range(len(board)):
            for col in range(len(board)):
                filled_score += (board[row][col] ** 2) * filled_score_map[row][col]
                if board[row][col] == 0:
                    empty_score = empty_score + 256 * empty_score_map[row][col]

        return 0.45 * filled_score + 0.55 * empty_score

    def is_goal(self, state) -> bool:
        return logic.game_state(state[0]) == 'lose'

    def copy_board(self, board):
        new_board = []
        for i in range(4):
            new_board.append([0] * 4)
        for i in range(4):
            for j in range(4):
                new_board[i][j] = board[i][j]
        return new_board


class ExpectimaxIndexPlayer(AbstractIndexPlayer):
    """Expectimax Index Player
    implement get_indices function according to Expectimax algorithm, the value is number between {2,4}.
    (you can add helper functions as you want)
    """

    def __init__(self):
        AbstractIndexPlayer.__init__(self)
        # fields used for error debugging & calculating average depth for q. 12 in dry assignment
        #self.depth_sums = 0
        #self.move_count = 0

    def get_indices(self, board, value, time_limit) -> (int, int):
        time_start = time.time()
        depth = 1
        min_val, min_move = self.ExpectimaxSearch((board, Turn.INDEX_PLAYER_TURN, value), Turn.INDEX_PLAYER_TURN, depth)
        curr_iter_time = time.time() - time_start
        node_ratio = 2 * 4 * 15
        prev_iter_time = curr_iter_time
        prev_prev_iter_time = curr_iter_time
        while node_ratio * prev_prev_iter_time < time_limit - (time.time() - time_start) - 0.01:
            depth += 1
            # print("(Comp) curr depth is: " + str(depth))
            iteration_start_time = time.time()
            last_good_indices = min_move
            val, min_move = self.ExpectimaxSearch((board, Turn.INDEX_PLAYER_TURN, value), Turn.INDEX_PLAYER_TURN, depth)
            if val == float('inf'):
                break
            if val == float('-inf'):
                min_move = last_good_indices
                break
            prev_prev_iter_time = prev_iter_time
            prev_iter_time = curr_iter_time
            next_iteration_max_time = node_ratio * prev_prev_iter_time
            curr_iter_time = time.time() - iteration_start_time
            # print(" prev prev iter time: " + str(prev_prev_iter_time) + " curr iter time: " + str(
            #    curr_iter_time) + " next max time: " + str(next_iteration_max_time))
        #self.move_count += 1
        #self.depth_sums += depth
        return min_move

    def ExpectimaxSearch(self, state, agent, depth):
        board = state[0]
        agentToMove = state[1]
        info_flag = state[2]

        if self.is_goal(state):  # can only reach goal as a result of index's move, so we return inf to reinforce this
            # move
            return float('inf'), None

        if depth == 0:
            return self.heuristic(state[0]), None

        if info_flag == 1:  # probability state appears before the index player's turn
            val2, _ = self.ExpectimaxSearch((self.copy_board(board), Turn.INDEX_PLAYER_TURN, 2), agent, depth - 1)
            val4, _ = self.ExpectimaxSearch((self.copy_board(board), Turn.INDEX_PLAYER_TURN, 4), agent, depth - 1)
            exp = 0.9 * val2 + 0.1 * val4
            return exp, None

        turn = agentToMove
        best_move = None
        if turn == agent:
            curr_max = float('-inf')
            for (i, j) in self.get_empty_indices(board):
                new_board = (self.copy_board(board))
                new_board[i][j] = info_flag
                new_state = (new_board, Turn.MOVE_PLAYER_TURN, 0)
                val, _ = self.ExpectimaxSearch(new_state, agent, depth - 1)
                if val >= curr_max:
                    curr_max = val
                    best_move = (i, j)
            return curr_max, best_move
        else:  # move player's turn
            cur_min = float('inf')
            for move in Move:
                new_board, valid, score = commands[move](self.copy_board(board))
                if valid:
                    new_state = (new_board, Turn.INDEX_PLAYER_TURN, 1)  # next state is probability state
                    val, _ = self.ExpectimaxSearch(new_state, agent, depth - 1)
                    if val <= cur_min:
                        cur_min = val
                        best_move = move
            return cur_min, best_move

    def get_empty_indices(self, board):
        empty = []
        for i in range(0, 4):
            for j in range(0, 4):
                if board[i][j] == 0:
                    empty.append((i, j))
        return empty

    def is_goal(self, state) -> bool:
        return logic.game_state(state[0]) == 'lose'

    def heuristic(self, board):
        filled_score_map = [[1, 0.5, 0.5, 1],
                            [0.5, 0.2, 0.2, 0.5],
                            [0.5, 0.2, 0.2, 0.5],
                            [1, 0.5, 0.5, 1]]
        empty_score_map = [[0, 0.5, 0.5, 0],
                           [0.5, 0.8, 0.8, 0.5],
                           [0.5, 0.8, 0.8, 0.5],
                           [0, 0.5, 0.5, 0]]
        filled_score = 0
        empty_score = 0
        for row in range(len(board)):
            for col in range(len(board)):
                filled_score += (board[row][col] ** 2) * filled_score_map[row][col]
                if board[row][col] == 0:
                    empty_score = empty_score + 256 * empty_score_map[row][col]

        return 0.45 * filled_score + 0.55 * empty_score

    def copy_board(self, board):
        new_board = []
        for i in range(4):
            new_board.append([0] * 4)
        for i in range(4):
            for j in range(4):
                new_board[i][j] = board[i][j]
        return new_board


# Tournament
class ContestMovePlayer(AbstractMovePlayer):
    """Contest Move Player,
    implement get_move function as you want to compete in the Tournament
    (you can add helper functions as you want)
    """

    def __init__(self):
        AbstractMovePlayer.__init__(self)
        self.clock = 0
        self.time_for_current_move = 0

    def get_move(self, board, time_limit) -> Move:
        self.clock = time.time()
        self.time_for_current_move = time_limit - 0.05
        depth = 1

        iteration_start_time = time.time()
        val, max_move = self.ContestMovePlayerSearch((board, Turn.MOVE_PLAYER_TURN), Turn.MOVE_PLAYER_TURN, depth,
                                                     float("-inf"), float("inf"))
        self.time_for_current_move -= (time.time() - iteration_start_time)

        while True:
            depth += 1
            last_good_move = max_move
            iteration_start_time = time.time()
            val, max_move = self.ContestMovePlayerSearch((board, Turn.MOVE_PLAYER_TURN), Turn.MOVE_PLAYER_TURN, depth,
                                                         float("-inf"), float("inf"))
            self.time_for_current_move -= (time.time() - iteration_start_time)

            if val == float('inf'):
                break
            if val == float('-inf'):
                max_move = last_good_move
                break
        return max_move

    def ContestMovePlayerSearch(self, state, agent, depth, alpha, beta):
        if self.no_more_time():
            return float("-inf"), None
        board = state[0]
        curr_turn = state[1]

        if self.is_goal(state):
            return float('-inf'), None
        if depth == 0:
            return self.heuristic(board), None

        turn = curr_turn
        best_move = None
        if turn == agent:
            cur_max = float("-inf")
            for move in Move:
                new_board, valid, score = commands[move](self.copy_board(board))
                if valid:
                    child_state = (new_board, Turn.INDEX_PLAYER_TURN)
                    v, _ = self.ContestMovePlayerSearch(child_state, agent, depth - 1, alpha, beta)
                    if v >= cur_max:
                        cur_max = v
                        best_move = move
                    alpha = max(cur_max, alpha)
                    if cur_max >= beta:
                        return float("inf"), move
            return cur_max, best_move
        else:
            cur_min = float("inf")
            for (i, j, v) in self.get_empty_indexes(board):
                new_board = self.copy_board(board)
                new_board[i][j] = v
                child_state = (new_board, Turn.MOVE_PLAYER_TURN)
                v, _ = self.ContestMovePlayerSearch(child_state, agent, depth - 1, alpha, beta)
                if v <= cur_min:
                    cur_min = v
                    best_move = (i, j, v)
                beta = min(cur_min, beta)
                if cur_min <= alpha:
                    return float("-inf"), (i, j, v)
            return cur_min, best_move

    def heuristic(self, board):
        filled_score_map = [[1, 0.5, 0.5, 1],
                            [0.5, 0.2, 0.2, 0.5],
                            [0.5, 0.2, 0.2, 0.5],
                            [1, 0.5, 0.5, 1]]
        empty_score_map = [[0, 0.5, 0.5, 0],
                           [0.5, 0.8, 0.8, 0.5],
                           [0.5, 0.8, 0.8, 0.5],
                           [0, 0.5, 0.5, 0]]
        filled_score = 0
        empty_score = 0
        for row in range(len(board)):
            for col in range(len(board)):
                filled_score += (board[row][col] ** 2) * filled_score_map[row][col]
                if board[row][col] == 0:
                    empty_score = empty_score + 256 * empty_score_map[row][col]

        return 0.45 * filled_score + 0.55 * empty_score

    def is_goal(self, state) -> bool:
        return logic.game_state(state[0]) == 'lose'

    def get_empty_indexes(self, board):
        empty = []
        for i in range(0, 4):
            for j in range(0, 4):
                if board[i][j] == 0:
                    empty.append((i, j, 2))
                    # empty.append((i, j, 4))
        return empty

    def copy_board(self, board):
        new_board = []
        for i in range(4):
            new_board.append([0] * 4)
        for i in range(4):
            for j in range(4):
                new_board[i][j] = board[i][j]
        return new_board

    def no_more_time(self):
        return (time.time() - self.clock) >= self.time_for_current_move
