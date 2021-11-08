#ifndef SELF_PLAY_H
#define SELF_PLAY_H

#include "mcts.hpp"
#include "node.hpp"

#include <chess/chess.hpp>
#include <vector>


struct SelfPlayWorker{
    size_t num_actions = 64*73;
    size_t max_iter = 800;
    size_t max_moves = 500;

    struct GameRow {
        chess::position state;
        std::vector<double> action_distribution;
    };

    void grind();
    void play_game();
    void print_row(const GameRow& row);

};

/*

# Finds the pawn taking move by quite high prob, good #
8r.bqkbnr
7ppppp.pp
6..n.....
5.....p..
4.....PP.
3........
2PPPPPK.P
1RNBQ.BNR
 abcdefgh
turn: black
white kingside castle: no
white queenside castle: no
black kingside castle: yes
black queenside castle: yes
halfmove clock: 0
fullmove number: 3

Best moves: (f5g4, 0.27375) - (c6d4, 0.18125) - (d7d6, 0.175) - 
*/

#endif // SELF_PLAY_H