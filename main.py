"""
- make the game
- 4x4
- arrows to move
- after move, a free square becomes a 2 or 4 (25%)
- print to terminal
"""

import math
import random

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
    # four 32-bit ints

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


class MultiPly4:
    def __init__(self, heuristic):
        self.heuristic = heuristic

    def choose_move(self, board):
        def inner_choose_move(board, depth):
            if depth == 0:
                return self.heuristic.score(board)
            best_score = None
            for direction in DIRECTIONS:
                if not move_is_valid(board, direction):
                    continue
                new_board = move(board, direction)
                if depth >= 1:
                    score = inner_choose_move(new_board, depth - 1)
                score = self.heuristic.score(new_board)
                if best_score is None or score > best_score:
                    best_score = score
            return best_score

        best_dir = None
        best_score = None
        for direction in DIRECTIONS:
            if not move_is_valid(board, direction):
                continue
            new_board = move(board, direction)
            score = inner_choose_move(new_board, 4)
            if best_score is None or score > best_score:
                best_dir = direction
                best_score = score
        return best_dir

    def name(self):
        return f"MultiPly4({self.heuristic.name()})"


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


def numeric_representation(board):
    row_bits = []
    for row in board:
        row_num = 0
        for i, elem in enumerate(row):
            row_num += (elem.bit_length() - 1) << (3 - i) * 8 if elem is not None else 0
        row_bits.append(row_num)
    return tuple(row_bits)


def create_rotate_rep():
    code = "def rotate_rep(rep): return ("
    rows = []
    for row_idx in range(4):
        row_components = []
        for col_idx in range(4):
            value = f"(((rep[{3 - col_idx}] >> {8 * (3 - row_idx)}) & 0xff) << {8 * (3 - col_idx)})"
            row_components.append(value)
        rows.append(" + ".join(row_components))
    code += ", ".join(rows)
    code += ")"

    return code


def left_rep(rep):
    result = []
    for row in rep:
        c1 = (row >> 24) & 0xFF
        c2 = (row >> 16) & 0xFF
        c3 = (row >> 8) & 0xFF
        c4 = row & 0xFF
        row = [value for value in (c1, c2, c3, c4) if value != 0]
        ret = 0
        shift = 24
        i = 0
        while i < len(row) - 1:
            a = row[i]
            b = row[i + 1]
            print(i, a, b)
            if a == b:
                value = a + 1
                i += 1
            else:
                value = a
            ret += value << shift
            shift -= 8
            i += 1
        result.append(ret)
    return tuple(result)


def print_numeric_representation(row_bits):
    for n in row_bits:
        s = ""
        for i in range(4):
            s += str((n >> (3 - i) * 8) & 0xFF) + ", "
        print(s)


print(create_rotate_rep())
exec(create_rotate_rep())
rep = numeric_representation(starting_board)
print_numeric_representation(rep)
lefted = left_rep(rep)
print_numeric_representation(lefted)

quit()


row_bits = numeric_representation(starting_board)
rotate_numeric_representation(row_bits)
print_numeric_representation(row_bits)

quit()

compare_strategies(
    [
        OnePly(HighestTile()),
        OnePly(EmptySpace()),
        OnePly(Combined()),
        MultiPly4(Combined()),
        Random(),
    ]
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
