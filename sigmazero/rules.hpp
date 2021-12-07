#ifndef RULES_HPP
#define RULES_HPP


#include <cstdint>

#include <chess/chess.hpp>
#include <torch/torch.h>

#include "sigmanet.hpp"


const int p1_piece_planes = 6;
const int p2_piece_planes = 6;
const int repetition_planes = 2;
const int feature_planes = p1_piece_planes + p2_piece_planes + repetition_planes;

const int color_planes = 1;
const int move_count_planes = 1;
const int p1_castling_planes = 2;
const int p2_castling_planes = 2;
const int no_progress_planes = 1;
const int constant_planes = color_planes + move_count_planes + p1_castling_planes + p2_castling_planes + no_progress_planes;

const int underpromotion_directions = 3;
const int underpromotion_pieces = 3;
const int underpromotion_actions = underpromotion_directions*underpromotion_pieces;

const int knight_directions = 8;
const int knight_actions = knight_directions;

const int sliding_directions = 8;
const int sliding_magnitudes = 7;
const int sliding_actions = sliding_directions*sliding_magnitudes;

const int actions_per_square = underpromotion_actions + knight_actions + sliding_actions;
const int num_actions = 64*actions_per_square;


int move_action(chess::move move, const chess::game& game);

torch::Tensor game_image(const chess::game& game, int history = 2);

sigmanet make_network(int history = 2, int filters = 128, int blocks = 10);


float material_value(const chess::game& game, chess::side side = chess::side_white);

float material_delta(const chess::game& game, chess::side side = chess::side_white);


#endif