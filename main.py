"""
- make the game
- 4x4
- arrows to move
- after move, a free square becomes a 2 or 4 (25%)
- print to terminal
"""

import random
from functools import partial


LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

DIRECTIONS = [LEFT, DOWN, UP, RIGHT]

starting_board = [
    [8, None, 8, None],
    [16, None, None, 16],
    [2, 2, 2, 2],
    [2, 2, 4, None],
]


def render(board):
    for row in board:
        s = ""
        for col in row:
            s += (f"{col}" if col else "_").rjust(4, " ")
        print(s)


def get_dir():
    while True:
        choice = input("> ")
        try:
            return {
                "u": UP,
                "r": RIGHT,
                "d": DOWN,
                "l": LEFT,
            }[choice]
        except KeyError:
            print("strange key!")
            pass


def shift(items):
    ret = []
    items_without_none = [item for item in items if item is not None]
    i = 0
    while i < len(items_without_none) - 1:
        a = items_without_none[i]
        b = items_without_none[i + 1]
        if a == b:
            ret.append(a + b)
            i += 1
        else:
            ret.append(a)
        i += 1

    if i == len(items_without_none) - 1:
        ret.append(items_without_none[-1])

    return ret + [None] * (len(items) - len(ret))


def left(board):
    return [shift(row) for row in board]


# for row_idx in range(len(board)-2, -1, -1):
#     for col_idx, col in enumerate(board[row_idx]):
#         if board[row_idx][col_idx] is None:
#             continue
#         for dest_row_idx in range(row_idx+1, len(board)):
#             us = board[row_idx][col_idx]
#             dest = board[dest_row_idx][col_idx]
#             # if dest == us:
#             # print(col_idx, dest_row_idx, dest)
#         # if row_idx > 0:


def rotate(board, rotations):
    for _ in range(rotations):
        board = list(map(list, list(zip(*board[::-1]))))
    return board


def spawn(board):
    free_spots = []
    for row_idx in range(len(board)):
        for col_idx in range(len(board[row_idx])):
            if board[row_idx][col_idx] is None:
                free_spots.append((row_idx, col_idx))
    row_idx, col_idx = random.choice(free_spots)
    board[row_idx][col_idx] = random.choice([2, 2, 2, 4])
    return board


def has_won(board):
    return any(col == 2048 for row in board for col in row)


def is_full(board):
    return all(col is not None for row in board for col in row)


def move(board, direction):
    return rotate(left(rotate(board, direction)), 4 - direction)


def has_lost(board):
    if not is_full(board):
        return False
    return all(move(board, direction) == board for direction in DIRECTIONS)


def win_or_lose(board):
    large_number = 10000000000
    if has_won(board):
        return large_number
    if has_lost(board):
        return -large_number
    return None


def empty_space(board):
    return sum(1 if col is None else 0 for row in board for col in row)


class EmptySpace:
    def score(self, board):
        wl = win_or_lose(board)
        if wl:
            return wl
        return empty_space(board)

    def name(self):
        return "EmptySpace"


class HighestTile:
    def score(self, board):
        wl = win_or_lose(board)
        if wl:
            return wl
        return highest_tile(board)

    def name(self):
        return "HighestTile"


class Combined:
    def score(self, board):
        wl = win_or_lose(board)
        if wl:
            return wl
        return empty_space(board) + highest_tile(board)

    def name(self):
        return "Combined"


class Random:
    def choose_move(self, board):
        return random.choice(DIRECTIONS)

    def name(self):
        return "Random"


class Human:
    def choose_move(self, board):
        render(board)
        return get_dir()

    def name(self):
        return "Human"


class OnePly:
    def __init__(self, heuristic):
        self.heuristic = heuristic

    def choose_move(self, board):
        best_dir = None
        best_score = None
        for direction in DIRECTIONS:
            if not move_is_valid(board, direction):
                continue
            new_board = move(board, direction)
            score = self.heuristic.score(new_board)
            if best_score is None or score > best_score:
                best_dir = direction
                best_score = score
        return best_dir

    def name(self):
        return f"OnePly({self.heuristic.name()})"


def highest_tile(board):
    return max(col for row in board for col in row if col is not None)


def move_is_valid(board, direction):
    new_board = move(board, direction)
    return new_board != board


def run_game(strategy):
    board = starting_board
    count = 0
    while not has_lost(board) and not has_won(board):
        direction = strategy.choose_move(board)
        prev_board = board
        board = move(board, direction)
        if prev_board != board:
            board = spawn(board)
            count += 1
    return highest_tile(board), count


def compare_strategies(strategies):
    NUMBER_OF_RUNS = 10
    for strategy in strategies:
        scores = []
        for _ in range(NUMBER_OF_RUNS):
            score, _ = run_game(strategy)
            scores.append(score)
        print(
            f"Stats for {strategy.name()}; max: {max(scores)}, avg: {sum(scores) / len(scores)}"
        )


compare_strategies(
    [OnePly(HighestTile()), OnePly(EmptySpace()), OnePly(Combined()), Random()]
)


def losing_board():
    count = 2
    result = []
    for _ in range(4):
        row = []
        for _ in range(4):
            row.append(count)
        result.append(row)
    return result
