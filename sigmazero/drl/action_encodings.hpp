#ifndef ACTION_ENCODINGS_H
#define ACTION_ENCODINGS_H

#include <chess/chess.hpp>
#include <unordered_map>
#include <utility>
#include <algorithm>
#include <map>


struct action_encodings {

    struct Action {
        std::string type;
        size_t pos;
        int v1, v2, v3;
    };
    static const std::string QUEEN_ACTION;
    static const std::string KNIGHT_ACTION;
    static const std::string UNDERPROMOTION_ACTION;
    
    static std::pair<int, int> knight_directions[8];
    static std::pair<int, int> queen_directions[8];
    static chess::piece underpromotions[3];
    static bool encoding_initialized;

    // Decoding map
    static Action actions[64*73];
    static size_t actions_flipped[64*73];

    // Encoding maps
    static std::map<std::tuple<size_t, int, int, size_t>, size_t> queen_actions;
    static std::map<std::tuple<size_t, int, int>, size_t> knight_actions;
    static std::map<std::tuple<size_t, int, size_t>, size_t> underpromotion_actions;

    static size_t cond_flip_action(const chess::position& state, size_t action);
    static void initialize_encoding_map();
    static size_t action_from_move(const chess::move& move);
    static chess::move move_from_action(const chess::position& state, size_t action_idx);

};


#endif // ACTION_ENCODINGS_H