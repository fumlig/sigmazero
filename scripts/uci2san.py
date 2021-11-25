#!/usr/bin/env python3

import chess
import sys

board = chess.Board()
moves = sys.argv[1:]
lines = False


for ply, lan in enumerate(moves):
    move = board.parse_uci(lan)
    san = board.san(move)
    board.push(move)

    pre = f"{int(ply/2 + 1)}. " if ply%2 == 0 else ""
    suf = " " if ply%2 == 1 else " "
    print(pre + san + suf, end="")
