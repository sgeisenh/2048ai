"""
- make the game
- 4x4
- arrows to move
- after move, a free square becomes a 2 or 4 (25%)
- print to terminal
"""

import random


LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

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


board = starting_board
while True:
    render(board)
    dir = get_dir()
    prev_board = board
    board = rotate(left(rotate(board, dir)), 4 - dir)
    if prev_board != board:
        board = spawn(board)
    # TODO: if dead board
