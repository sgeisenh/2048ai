A win is when a 2048 tile appears on the board.

A dead board is when no directions can be played (and the board is full).
Board being full is a nice heuristic.

Need some sort of board heuristic.

First strategy:
Choose the next move (depth of 1).
Make all 4 moves, measure them and choose the best.

Iteration:
Choose a particular depth and determine which move will lead to the best move
tree.

Let's take a weighted average across the possible board states.
"expectimax"
