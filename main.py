import argparse
import random
from enum import Enum
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from time import perf_counter
from typing import (
    Type,
    Collection,
    Final,
    Iterable,
    Iterator,
    Protocol,
    Sequence,
)

from tqdm import tqdm


# TYPES!


class Direction(Enum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


Board = list[list[int]]


class Representation(Protocol):
    def __repr__(self) -> str:
        ...

    def __eq__(self, other: object, /) -> bool:
        ...

    def __iter__(self) -> Iterator[int]:
        ...

    @classmethod
    def from_board(cls, board: Board, /) -> "Representation":
        ...

    def rotate(self) -> "Representation":
        ...

    def shift_left(self) -> "Representation":
        ...

    def spawn(self) -> "Representation":
        ...

    def highest_tile(self) -> int:
        return max(self)

    def num_empties(self) -> int:
        return sum(1 if elem else 0 for elem in self)

    def free_spots(self) -> list[tuple[int, int]]:
        return [(idx // 4, idx % 4) for idx, elem in enumerate(self) if elem == 0]

    def has_lost(self) -> bool:
        if any(value == 0 for value in self):
            return False
        return all(self.move(direction) == self for direction in Direction)

    def has_won(self) -> bool:
        return self.highest_tile() == 2048

    def rotate_n(self, n: int, /) -> "Representation":
        result = self
        for _ in range(n):
            result = result.rotate()
        return result

    def move(self, direction: Direction, /) -> "Representation":
        value = direction.value
        return self.rotate_n(value).shift_left().rotate_n(4 - value)

    def possible_moves(self) -> dict[Direction, "Representation"]:
        return {
            direction: new_rep
            for direction in Direction
            if (new_rep := self.move(direction)) != self
        }


class Strategy(Protocol):
    def choose_move(self, rep: Representation, /) -> Representation | None:
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class Heuristic(Protocol):
    def score(self, rep: Representation, /) -> int:
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


# Common constants and helpers.


STARTING_BOARD: Final[Board] = [
    [2, 0, 0, 0],
    [0, 2, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
]


SPAWN_POWERS: tuple[int, int, int, int] = (1, 1, 1, 2)


def get_new_tile_power() -> int:
    return random.choice(SPAWN_POWERS)


def move_if_valid(rep: Representation, direction: Direction) -> Representation | None:
    new_rep = rep.move(direction)
    return new_rep if new_rep != rep else None


# Naiive board representation.


def shift(row: Collection[int]) -> list[int]:
    ret = []
    without_none = [entry for entry in row if entry != 0]
    i = 0
    while i < len(without_none) - 1:
        left = without_none[i]
        right = without_none[i + 1]
        if left == right:
            ret.append(2 * left)
            i += 1
        else:
            ret.append(left)
        i += 1

    if i < len(without_none):
        ret.append(without_none[-1])

    return ret + [0] * (len(row) - len(ret))


class NaiiveRep(Representation):
    __slots__ = ("_board",)

    def __init__(self, board: Board) -> None:
        self._board = board

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NaiiveRep) and self._board == other._board

    def __iter__(self) -> Iterator[int]:
        return (col for row in self._board for col in row)

    def __repr__(self) -> str:
        result = []
        for row in self._board:
            s = ""
            for col in row:
                s += (f"{col}" if col else "_").rjust(4, " ")
            result.append(s)
        return "\n".join(result)

    @classmethod
    def from_board(cls, board: Board) -> "NaiiveRep":
        return cls(board)

    def rotate(self) -> "NaiiveRep":
        return NaiiveRep(list(map(list, zip(*self._board[::-1]))))

    def shift_left(self) -> "NaiiveRep":
        return NaiiveRep([shift(row) for row in self._board])

    # Note that this mutates the current board. But at least it is pretty fast.
    def spawn(self) -> "NaiiveRep":
        free_spots = self.free_spots()
        if not free_spots:
            print(self)
            raise ValueError("No free spots?")
        row_idx, col_idx = random.choice(self.free_spots())
        self._board[row_idx][col_idx] = 2 ** get_new_tile_power()
        return self


# Code to generate `NumericRep.rotate()`.


def create_col_component(row_idx: int, col_idx: int) -> str:
    return (
        f"(((rep[{3 - col_idx}] >> {8 * (3 - row_idx)}) & 0xff) << {8 * (3 - col_idx)})"
    )


def create_row_component(row_idx: int) -> str:
    return " + ".join(create_col_component(row_idx, col_idx) for col_idx in range(4))


def create_numeric_rotate() -> str:
    arg_sep = f',\n{8 * " "}'
    return f"""\
def rotate(self) -> "NumericRep":
    rep = self._numbers
    return NumericRep(
        (
            {arg_sep.join(create_row_component(row_idx) for row_idx in range(4))}
        )
    )
"""


# Numeric representation.
# Would maybe be more efficient to store left elements in low-order bits?


class NumericRep(Representation):
    __slots__ = ("_numbers",)

    def __init__(self, numbers: tuple[int, ...]) -> None:
        self._numbers = numbers

    def __iter__(self) -> Iterator[int]:
        for n in self._numbers:
            for i in range(4):
                power = (n >> (3 - i) * 8) & 0xFF
                yield 2**power if power != 0 else 0

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NumericRep) and self._numbers == other._numbers

    def __repr__(self) -> str:
        result = []
        for n in self._numbers:
            s = ""
            for i in range(4):
                end = ", " if i != 3 else ""
                s += str((n >> (3 - i) * 8) & 0xFF) + end
            result.append(s)
        return "\n".join(result)

    @classmethod
    def from_board(cls, board: Board) -> "NumericRep":
        row_bits = []
        for row in board:
            row_num = 0
            for i, elem in enumerate(row):
                row_num += (elem.bit_length() - 1) << (3 - i) * 8 if elem != 0 else 0
            row_bits.append(row_num)
        return cls(tuple(row_bits))

    # Generated by `create_numeric_rep` above.
    def rotate(self) -> "NumericRep":
        rep = self._numbers
        return NumericRep(
            (
                (((rep[3] >> 24) & 0xFF) << 24)
                + (((rep[2] >> 24) & 0xFF) << 16)
                + (((rep[1] >> 24) & 0xFF) << 8)
                + (((rep[0] >> 24) & 0xFF) << 0),
                (((rep[3] >> 16) & 0xFF) << 24)
                + (((rep[2] >> 16) & 0xFF) << 16)
                + (((rep[1] >> 16) & 0xFF) << 8)
                + (((rep[0] >> 16) & 0xFF) << 0),
                (((rep[3] >> 8) & 0xFF) << 24)
                + (((rep[2] >> 8) & 0xFF) << 16)
                + (((rep[1] >> 8) & 0xFF) << 8)
                + (((rep[0] >> 8) & 0xFF) << 0),
                (((rep[3] >> 0) & 0xFF) << 24)
                + (((rep[2] >> 0) & 0xFF) << 16)
                + (((rep[1] >> 0) & 0xFF) << 8)
                + (((rep[0] >> 0) & 0xFF) << 0),
            )
        )

    def shift_left(self) -> "NumericRep":
        result = []
        for row in self._numbers:
            c1 = (row >> 24) & 0xFF
            c2 = (row >> 16) & 0xFF
            c3 = (row >> 8) & 0xFF
            c4 = row & 0xFF
            row_list = [value for value in (c1, c2, c3, c4) if value != 0]
            ret = 0
            shift = 24
            i = 0
            while i < len(row_list) - 1:
                a = row_list[i]
                b = row_list[i + 1]
                if a == b:
                    value = a + 1
                    i += 1
                else:
                    value = a
                ret += value << shift
                shift -= 8
                i += 1
            if i == len(row_list) - 1:
                ret += row_list[-1] << shift
            result.append(ret)
        return NumericRep(tuple(result))

    def spawn(self) -> "NumericRep":
        free_spots = self.free_spots()
        if not free_spots:
            print(self)
            raise ValueError("No free spots?")
        row, col = random.choice(self.free_spots())
        to_add = get_new_tile_power() << (3 - col) * 8
        return NumericRep(
            (
                self._numbers[0] if row != 0 else self._numbers[0] + to_add,
                self._numbers[1] if row != 1 else self._numbers[1] + to_add,
                self._numbers[2] if row != 2 else self._numbers[2] + to_add,
                self._numbers[3] if row != 3 else self._numbers[3] + to_add,
            )
        )


# Heuristics.


def win_or_lose(rep: Representation) -> int | None:
    large_number = 10000000000
    if rep.has_won():
        return large_number
    if rep.has_lost():
        return -large_number
    return None


class EmptySpace(Heuristic):
    def score(self, rep: Representation) -> int:
        wl = win_or_lose(rep)
        if wl:
            return wl
        return rep.num_empties()


class HighestTile(Heuristic):
    def score(self, rep: Representation) -> int:
        wl = win_or_lose(rep)
        if wl:
            return wl
        return rep.highest_tile()


class Combined(Heuristic):
    def score(self, rep: Representation) -> int:
        wl = win_or_lose(rep)
        if wl:
            return wl
        return rep.num_empties() + rep.highest_tile()


# Strategies.


class Random(Strategy):
    def choose_move(self, rep: Representation) -> Representation | None:
        possible_moves = list(rep.possible_moves().values())
        if not possible_moves:
            return None
        return random.choice(possible_moves)


class Human(Strategy):
    def choose_move(self, rep: Representation) -> Representation | None:
        possible_moves = rep.possible_moves()
        print(f"Current board: \n{rep}\n")
        while True:
            # TODO: refactor this to await some sort of arrow key press?
            choice = input("> ")
            try:
                direction = {
                    "u": Direction.UP,
                    "r": Direction.RIGHT,
                    "d": Direction.DOWN,
                    "l": Direction.LEFT,
                }[choice]
            except KeyError:
                print("Strange key! Try again.")
                continue
            try:
                return possible_moves[direction]
            except KeyError:
                print(
                    f"Sorry, you can't move {direction} on the current board, try a different direction."
                )


class OnePly(Strategy):
    def __init__(self, heuristic: Heuristic) -> None:
        self._heuristic = heuristic

    def __repr__(self) -> str:
        return f"OnePly({self._heuristic!r})"

    def choose_move(self, rep: Representation) -> Representation | None:
        return max(
            (
                new_rep
                for direction in Direction
                if (new_rep := move_if_valid(rep, direction)) is not None
            ),
            key=self._heuristic.score,  # type: ignore
            default=None,
        )


class MultiPly4(Strategy):
    def __init__(self, heuristic: Heuristic) -> None:
        self._heuristic = heuristic

    def __repr__(self) -> str:
        return f"MultiPly4({self._heuristic!r})"

    def choose_move(self, rep: Representation) -> Representation | None:
        def score(rep: Representation, depth: int) -> int | None:
            if depth == 0:
                return self._heuristic.score(rep)
            return max(
                (
                    score(new_rep, depth - 1) or 0
                    for direction in Direction
                    if (new_rep := move_if_valid(rep, direction)) is not None
                ),
                default=None,
            )

        return max(
            (
                new_rep
                for direction in Direction
                if (new_rep := move_if_valid(rep, direction)) is not None
            ),
            key=lambda rep: score(rep, 4) or 0,  # type: ignore
            default=None,
        )


@dataclass
class GameResult:
    highest_tile: int
    number_of_moves: int
    wall_time_s: float


def run_game(strategy: Strategy, rep_cls: Type[Representation]) -> GameResult:
    start_time = perf_counter()
    rep = rep_cls.from_board(STARTING_BOARD)
    count = 0
    while not rep.has_lost() and not rep.has_won():
        maybe_rep = strategy.choose_move(rep)
        assert maybe_rep is not None, "We must have lost?"
        rep = maybe_rep.spawn()
        count += 1
    end_time = perf_counter()
    return GameResult(rep.highest_tile(), count, end_time - start_time)


@dataclass
class StrategyStats:
    strategy: str
    rep_cls: str
    avg_time: float
    max_score: int
    avg_score: float


def compare_strategies(
    strategies: Iterable[Strategy],
    rep_classes: Sequence[Type[Representation]],
    iterations: int = 1000,
    in_process: bool = False,
) -> list[StrategyStats]:
    stats = []
    for strategy in strategies:
        for rep_cls in rep_classes:
            print(
                f"Running strategy {strategy} with representation {rep_cls.__name__}..."
            )
            run = partial(run_game, strategy, rep_cls)
            if in_process:
                results = [run() for _ in tqdm(range(iterations), total=iterations)]
            else:
                with ProcessPoolExecutor() as executor:
                    futures = [executor.submit(run) for _ in range(iterations)]
                    results = [
                        future.result()
                        for future in tqdm(as_completed(futures), total=iterations)
                    ]
            avg_time = sum(result.wall_time_s for result in results) / len(results)
            max_score = max(result.highest_tile for result in results)
            avg_score = sum(result.highest_tile for result in results) / len(results)
            stats.append(
                StrategyStats(
                    repr(strategy), repr(rep_cls), avg_time, max_score, avg_score
                )
            )
    return stats


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_numeric_rotate", action="store_true")
    parser.add_argument("--in_process", action="store_true")
    args = parser.parse_args(argv)
    if args.gen_numeric_rotate:
        print(create_numeric_rotate())
        return 0

    for stats in compare_strategies(
        [
            Random(),
            OnePly(HighestTile()),
            OnePly(EmptySpace()),
            OnePly(Combined()),
            MultiPly4(HighestTile()),
            MultiPly4(EmptySpace()),
            MultiPly4(Combined()),
        ],
        [
            NaiiveRep,
            NumericRep,
        ],
        iterations=10 if args.in_process else 1000,
        in_process=args.in_process,
    ):
        print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
