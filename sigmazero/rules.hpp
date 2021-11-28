#ifndef RULES_HPP
#define RULES_HPP


#include <cstdint>

#include <chess/chess.hpp>
#include <torch/torch.h>

#include "sigmanet.hpp"


const int feature_planes = 6 + 6 + 2; // p1 piece, p2 piece, repetitions
const int constant_planes = 1 + 1 + 2 + 2 + 1; // colour, total move count, p1 castling, p2 castling, no-progress count

const int sliding_actions = 7*8; // 8 directions, 7 squares in each direction
const int knight_actions = 8; // 8 directions
const int underpromotion_actions = 3*3; // 3 directions, 3 underpromotions pieces

const int actions_per_square = sliding_actions + knight_actions + underpromotion_actions;
const int num_actions = 64*actions_per_square;


int move_action(chess::move move, chess::side turn = chess::side_white);

chess::move action_move(int action, chess::side turn = chess::side_white);

torch::Tensor game_image(const chess::game& game, int history = 2);

sigmanet make_network(int history = 2);



#endif