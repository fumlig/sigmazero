#include "action_encodings.hpp"
#include <chess/chess.hpp>
#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <utility>
#include <cassert>
#include <set>


bool action_encodings::encoding_initialized = false;

// direction index -> X-Y coords
std::pair<int, int> action_encodings::knight_directions[8] = {
    std::make_pair(1,2),
    std::make_pair(2,1),
    std::make_pair(2,-1),
    std::make_pair(1,-2),
    std::make_pair(-1,-2),
    std::make_pair(-2,-1),
    std::make_pair(-2,1),
    std::make_pair(-1,2)
};
// direction index -> X-Y coords
std::pair<int, int> action_encodings::queen_directions[8] = {
    std::make_pair(0,1),
    std::make_pair(1,1),
    std::make_pair(1,0),
    std::make_pair(1,-1),
    std::make_pair(0,-1),
    std::make_pair(-1,-1),
    std::make_pair(-1,0),
    std::make_pair(-1,1)
};

chess::piece action_encodings::underpromotions[3] = {
    chess::piece_knight,
    chess::piece_bishop,
    chess::piece_rook
};
const std::string action_encodings::QUEEN_ACTION = "Queen";
const std::string action_encodings::KNIGHT_ACTION = "Knight";
const std::string action_encodings::UNDERPROMOTION_ACTION = "Underpromotion";

action_encodings::Action action_encodings::actions[64*73];
size_t action_encodings::actions_flipped[64*73];

std::map<std::tuple<size_t, int, int, size_t>, size_t> action_encodings::queen_actions;
std::map<std::tuple<size_t, int, int>, size_t> action_encodings::knight_actions;
std::map<std::tuple<size_t, int, size_t>, size_t> action_encodings::underpromotion_actions;


size_t action_encodings::cond_flip_action(const chess::position& state, size_t action){
    chess::side p1 = state.get_turn();
    bool flip = p1 == chess::side_black;
    return flip ? actions_flipped[action] : action;
}


void action_encodings::initialize_encoding_map(){
    size_t action = 0;
    for (size_t pos = 0; pos < 64; pos++) {
        for (int dir = 0; dir < 8; dir++) {
            for (int magnitude = 1; magnitude <= 7; magnitude++) {
                auto [dx, dy] = queen_directions[dir];
                queen_actions[std::make_tuple(pos, dx, dy, magnitude)] = action;
                actions[action++] = {QUEEN_ACTION, pos, dx, dy, magnitude};
            }
            auto [dx, dy] = knight_directions[dir];
            knight_actions[std::make_tuple(pos, dx, dy)] = action;
            actions[action++] = {KNIGHT_ACTION, pos, dx, dy};
        }

        for (int dx = -1; dx <= 1; dx++) {
            for (int u = 0; u < 3; u++) {
                underpromotion_actions[std::make_tuple(pos, dx, u)] = action;
                actions[action++] = {UNDERPROMOTION_ACTION, pos, dx, u};
            }
        }
    }
    // Do flipped mappings
    action = 0;
    for (size_t pos = 0; pos < 64; pos++) {
        int y = pos / 8;
        int x = pos % 8;
        y = 7 - y;
        size_t flipped_pos = y * 8 + x;
        for (int dir = 0; dir < 8; dir++) {
            for (int magnitude = 1; magnitude <= 7; magnitude++) {
                auto [dx, dy] = queen_directions[dir];
                actions_flipped[action++] = queen_actions[std::make_tuple(flipped_pos, dx, -dy, magnitude)];
            }
            auto [dx, dy] = knight_directions[dir];
            actions_flipped[action++] = knight_actions[std::make_tuple(flipped_pos, dx, -dy)];
        }

        for (int dx = -1; dx <= 1; dx++) {
            for (int u = 0; u < 3; u++) { 
                actions_flipped[action++] = underpromotion_actions[std::make_tuple(flipped_pos, dx, u)];
            }
        }
    }
    encoding_initialized = true;
}


size_t action_encodings::action_from_move(const chess::move& move) {
    if (!encoding_initialized) {
        initialize_encoding_map();
    }
    int delta_x = (move.to % 8) - (move.from % 8);
    int delta_y = (move.to / 8) - (move.from / 8);
    size_t pos = (size_t) move.from;
    // UNDER PROMOTE
    if (move.promote != chess::piece_none && move.promote != chess::piece_queen) {
        size_t piece_idx = 0;
        switch(move.promote){
            case chess::piece_knight: piece_idx = 0; break;
            case chess::piece_bishop: piece_idx = 1; break;
            case chess::piece_rook: piece_idx = 2; break;
            default: std::cerr << "Underpromotion is not valid" << std::endl;
        }
        return underpromotion_actions[std::make_tuple(pos, delta_x, piece_idx)];
    }
    // KNIGHT
    if (std::abs(delta_x) != std::abs(delta_y) && delta_x != 0 && delta_y != 0){
        return knight_actions[std::make_tuple(pos, delta_x, delta_y)];
    }
    int dir_x = delta_x != 0 ? delta_x / std::abs(delta_x) : 0;
    int dir_y = delta_y != 0 ? delta_y / std::abs(delta_y) : 0;
    size_t magnitude = std::max(std::abs(delta_x), std::abs(delta_y));
    return queen_actions[std::make_tuple(pos, dir_x, dir_y, magnitude)];
}

chess::move action_encodings::move_from_action(const chess::position& state, size_t action_idx) {
    if (!encoding_initialized) {
        initialize_encoding_map();
    }
    Action action = actions[action_idx];
    chess::square from = static_cast<chess::square>(action.pos);
    size_t x = action.pos % 8;
    size_t y = action.pos / 8;
    if (action.type == UNDERPROMOTION_ACTION) {
        int dx = action.v1;
        size_t u = action.v2;
        size_t y_promotion = y == 1 ? 0 : 7;
        chess::square to = static_cast<chess::square>(x + dx + (y_promotion)*8);
        chess::piece promote = underpromotions[u];
        chess::move move{from, to, promote};
        return move;
    }
    int dx = action.v1;
    int dy = action.v2;
    if (action.type == KNIGHT_ACTION) {
        chess::square to = static_cast<chess::square>(x + dx + (y + dy)*8);
        chess::move move{from, to};
        return move;
    }
    size_t magnitude = action.v3;
    dx *= magnitude;
    dy *= magnitude;
    int to_y = y + dy;
    chess::square to = static_cast<chess::square>(x + dx + to_y*8);
    chess::move move{from, to};
    auto [side, piece] = state.get_board().get(move.from);
    if (piece == chess::piece_pawn && (to_y == 0 || to_y == 7)) {
        move.promote = chess::piece::piece_queen;
    }
    return move;
}