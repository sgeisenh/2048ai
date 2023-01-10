"""
- make the game
- 4x4
- arrows to move
- after move, a free square becomes a 2 or 4 (25%)
- print to terminal
"""

import dis
import math
import random
from abc import ABC, abstractmethod

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3


def direction_to_str(direction):
    return {
        0: "LEFT",
        1: "DOWN",
        2: "RIGHT",
        3: "UP",
    }[direction]


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


def get_new_tile_power():
    return random.choice([1, 1, 1, 2])


def spawn(board):
    free_spots = []
    for row_idx in range(len(board)):
        for col_idx in range(len(board[row_idx])):
            if board[row_idx][col_idx] is None:
                free_spots.append((row_idx, col_idx))
    row_idx, col_idx = random.choice(free_spots)
    board[row_idx][col_idx] = 2 ** get_new_tile_power()
    return board


def has_won(board):
    return any(col == 2048 for row in board for col in row)


def is_full(board):
    return all(col is not None for row in board for col in row)


def move(board, direction):
    return rotate(left(rotate(board, direction)), 4 - direction)


# TODO add this optimization to numeric rep
def has_lost(board):
    if not is_full(board):
        return False
    return all(move(board, direction) == board for direction in DIRECTIONS)


def win_or_lose(rep):
    large_number = 10000000000
    if rep.has_won():
        return large_number
    if rep.has_lost():
        return -large_number
    return None


class EmptySpace:
    def score(self, rep):
        wl = win_or_lose(rep)
        if wl:
            return wl
        return rep.num_empties()

    def name(self):
        return "EmptySpace"


class HighestTile:
    def score(self, rep):
        wl = win_or_lose(rep)
        if wl:
            return wl
        return rep.highest_tile()

    def name(self):
        return "HighestTile"


class Combined:
    def score(self, rep):
        wl = win_or_lose(rep)
        if wl:
            return wl
        return rep.num_empties() + rep.highest_tile()

    def name(self):
        return "Combined"


class Random:
    def choose_move(self, rep):
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

    def choose_move(self, rep):
        best_dir = None
        best_score = None
        for direction in DIRECTIONS:
            if not move_is_valid(rep, direction):
                continue
            new_rep = rep.move(direction)
            score = self.heuristic.score(new_rep)
            if best_score is None or score > best_score:
                best_dir = direction
                best_score = score
        return best_dir

    def name(self):
        return f"OnePly({self.heuristic.name()})"


class MultiPly4:
    def __init__(self, heuristic):
        self.heuristic = heuristic

    def choose_move(self, rep):
        def inner_choose_move(rep, depth):
            if depth == 0:
                return self.heuristic.score(rep)
            best_score = None
            for direction in DIRECTIONS:
                if not move_is_valid(rep, direction):
                    continue
                new_rep = rep.move(direction)
                if depth >= 1:
                    score = inner_choose_move(new_rep, depth - 1)
                score = self.heuristic.score(new_rep)
                if best_score is None or score > best_score:
                    best_score = score
            return best_score

        best_dir = None
        best_score = None
        for direction in DIRECTIONS:
            if not move_is_valid(rep, direction):
                continue
            new_rep = rep.move(direction)
            score = inner_choose_move(new_rep, 4)
            if best_score is None or score > best_score:
                best_dir = direction
                best_score = score
        return best_dir

    def name(self):
        return f"MultiPly4({self.heuristic.name()})"


def highest_tile(board):
    return max(col for row in board for col in row if col is not None)


def move_is_valid(rep, direction):
    new_rep = rep.move(direction)
    return new_rep != rep


def run_game(strategy, Rep):
    rep = Rep.from_board(starting_board)
    count = 0
    while not rep.has_lost() and not rep.has_won():
        direction = strategy.choose_move(rep)
        prev_rep = rep
        rep = rep.move(direction)
        if rep != prev_rep:
            rep = rep.spawn()
            count += 1
    return rep.highest_tile(), count


class Representation(ABC):
    @abstractmethod
    def move(self, direction):
        ...

    @abstractmethod
    def __eq__(self, other):
        ...

    @abstractmethod
    def highest_tile(self):
        ...

    @abstractmethod
    def num_empties(self):
        ...

    @abstractmethod
    def spawn(self):
        ...

    def has_lost(self):
        if any(value == 0 for value in self):
            return False
        return all(self.move(direction) == self for direction in DIRECTIONS)

    def has_won(self):
        return self.highest_tile() == 2048


def numeric_representation(board):
    row_bits = []
    for row in board:
        row_num = 0
        for i, elem in enumerate(row):
            row_num += (elem.bit_length() - 1) << (3 - i) * 8 if elem is not None else 0
        row_bits.append(row_num)
    return tuple(row_bits)


def print_numeric_representation(row_bits):
    result = []
    for n in row_bits:
        s = ""
        for i in range(4):
            end = ", " if i != 3 else ""
            s += str((n >> (3 - i) * 8) & 0xFF) + end
        result.append(s)
    return "\n".join(result)


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


exec(create_rotate_rep())
# print(dis.dis(rotate_rep))
# quit()


def rotate_rep_times(rep, num):
    result = rep
    for _ in range(num):
        result = rotate_rep(result)
    return result


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
            if a == b:
                value = a + 1
                i += 1
            else:
                value = a
            ret += value << shift
            shift -= 8
            i += 1
        if i == len(row) - 1:
            ret += row[-1] << shift
        result.append(ret)
    return tuple(result)


class NumericRep(Representation):
    __slots__ = ("_numbers",)

    def __init__(self, numbers):
        self._numbers = numbers

    def __repr__(self):
        return print_numeric_representation(self._numbers)

    @classmethod
    def from_board(cls, board):
        return cls(numeric_representation(board))

    def move(self, direction):
        return NumericRep(
            rotate_rep_times(
                left_rep(rotate_rep_times(self._numbers, direction)), 4 - direction
            )
        )

    def __eq__(self, other):
        return self._numbers == other._numbers

    def highest_tile(self):
        return 2 ** max(self)

    def num_empties(self):
        return sum(1 if value == 0 else 0 for value in self)

    def __iter__(self):
        return (n >> (3 - i) * 8 & 0xFF for n in self._numbers for i in range(4))

    def spawn(self):
        possible = []
        for idx, value in enumerate(self):
            if value == 0:
                row = idx // 4
                col = idx % 4
                possible.append((row, col))
        row, col = random.choice(possible)
        to_add = get_new_tile_power() << (3 - col) * 8
        return NumericRep(
            (
                self._numbers[0] if row != 0 else self._numbers[0] + to_add,
                self._numbers[1] if row != 1 else self._numbers[1] + to_add,
                self._numbers[2] if row != 2 else self._numbers[2] + to_add,
                self._numbers[3] if row != 3 else self._numbers[3] + to_add,
            )
        )


def compare_strategies(strategies):
    NUMBER_OF_RUNS = 10
    for strategy in strategies:
        scores = []
        for _ in range(NUMBER_OF_RUNS):
            score, _ = run_game(strategy, NumericRep)
            scores.append(score)
        print(
            f"Stats for {strategy.name()}; max: {max(scores)}, avg: {sum(scores) / len(scores)}"
        )


compare_strategies(
    [
        Random(),
        OnePly(HighestTile()),
        OnePly(EmptySpace()),
        OnePly(Combined()),
        MultiPly4(Combined()),
    ]
)
