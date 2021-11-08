#ifndef NETWORK_H
#define NETWORK_H

#include <chess/chess.hpp>
#include <unordered_map>
#include <utility>
#include <algorithm>
#include <map>
namespace mcts {

struct Network {
    struct Evaluation {
        double value;
        std::unordered_map<size_t, double> action_probabilities;
    };

    struct Action {
        std::string type;
        size_t pos;
        int v1, v2, v3;
    };
    static const std::string QUEEN_ACTION;
    static const std::string KNIGHT_ACTION;
    static const std::string UNDERPROMOTION_ACTION;
    
    Evaluation evaluate(chess::position& state) const;

    static std::pair<int, int> knight_directions[8];
    static std::pair<int, int> queen_directions[8];
    static chess::piece underpromotions[3];
    static bool encoding_initialized;
    // Decoding map
    static Action actions[64*73];
    // Encoding maps
    static std::map<std::tuple<size_t, int, int, size_t>, size_t> queen_actions;
    static std::map<std::tuple<size_t, int, int>, size_t> knight_actions;
    static std::map<std::tuple<size_t, int, size_t>, size_t> underpromotion_actions;
    static void initialize_encoding_map();
    // SHOULD EXIST SIMPLER WAY, DO LATER
    static size_t action_from_move(const chess::move& move);
    static chess::move move_from_action(const chess::position& state, size_t action);

};

}

#endif // NETWORK_H