from GameWrapper import *
from logic import *
from submission import *
import copy


class KeyBoardGame(GameGrid):
    def __init__(self):
        GameGrid.__init__(self)
        self.master.bind("<Key>", self.key_down)

        self.commands = {c.KEY_UP: logic.up, c.KEY_DOWN: logic.down,
                         c.KEY_LEFT: logic.left, c.KEY_RIGHT: logic.right,
                         c.KEY_UP_ALT: logic.up, c.KEY_DOWN_ALT: logic.down,
                         c.KEY_LEFT_ALT: logic.left, c.KEY_RIGHT_ALT: logic.right,
                         c.KEY_H: logic.left, c.KEY_L: logic.right,
                         c.KEY_K: logic.up, c.KEY_J: logic.down}

    def key_down(self, event):
        key = repr(event.char)
        if key in self.commands:
            self.matrix, done, score = self.commands[repr(event.char)](self.matrix)
            self.total_score += score
            #print(f'score + {score}')
            if done:
                self.matrix = logic.add_two(self.matrix)
                self.update_grid_cells()

                if logic.game_state(self.matrix) == 'lose':
                    self.grid_cells[1][1].configure(text="Game", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                    self.grid_cells[1][2].configure(text="Over", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                    max_val = max(max(row) for row in self.matrix)
                    print(f'Game Over with max value = {max_val} and Total score = {self.total_score}')

    def run_game(self):
        self.mainloop()


class CustomGame(GameGrid):
    def __init__(self, move_player, index_player, move_time_limit, random_value=False):
        GameGrid.__init__(self)
        self.commands = {Move.UP: logic.up, Move.DOWN: logic.down,
                         Move.LEFT: logic.left, Move.RIGHT: logic.right}
        self.move_player = move_player
        self.index_player = index_player
        self.move_time_limit = move_time_limit
        self.random_value = random_value
        self.curr_agent = Turn.MOVE_PLAYER_TURN
        self.update()

    def run_game(self):
        while True:
            if self.curr_agent == Turn.MOVE_PLAYER_TURN:
                start = time.time()
                move = self.move_player.get_move(copy.deepcopy(self.matrix), self.move_time_limit)
                end = time.time()
                time_diff = end - start
                #print(time_diff)
                if time_diff > self.move_time_limit:
                    print(f'Time up for player {Turn.MOVE_PLAYER_TURN}')
                    break

                if move is None:
                    print("illegal move")
                    break

                #print(move)
                self.matrix, done, score = self.commands[move](self.matrix)
                if not done:
                    print("illegal move")
                    break

                self.total_score += score
                #print(f'score + {score}')
                self.curr_agent = Turn.INDEX_PLAYER_TURN

            elif self.curr_agent == Turn.INDEX_PLAYER_TURN:

                value = 2
                if self.random_value:
                    value = gen_two_or_four(PROBABILITY)

                start = time.time()
                indices = self.index_player.get_indices(copy.deepcopy(self.matrix), value, self.move_time_limit)
                end = time.time()
                time_diff = end - start

                if time_diff > self.move_time_limit:
                    print(f'Time up for player {Turn.INDEX_PLAYER_TURN}')
                    break
                if indices is None:
                    print("illegal indices")
                    break

                i, j = indices
                if i < 0 or j < 0 or i >= c.GRID_LEN or j >= c.GRID_LEN or self.matrix[i][j] != 0:
                    print("illegal indices")
                    break

                self.matrix[i][j] = value
                self.curr_agent = Turn.MOVE_PLAYER_TURN

            self.update_grid_cells()
            if logic.game_state(self.matrix) == 'lose':
                break

            self.update()

        self.grid_cells[1][1].configure(text="Game", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
        self.grid_cells[1][2].configure(text="Over", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
        self.update()

        max_val = max(max(row) for row in self.matrix)
        print(f'Game Over with max value = {max_val} and Total score = {self.total_score}')
        # See the final board before it disappears
        time.sleep(3)
