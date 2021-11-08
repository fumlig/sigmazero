#include "self_play.hpp"
#include "mcts.hpp"

#include "node.hpp"
#include "network.hpp"
#include <iostream>
#include <memory>
#include <utility>
#include <algorithm>

#include <vector>


void SelfPlayWorker::grind(){
    while(true) {
        play_game();
    }

}

void SelfPlayWorker::play_game(){
    chess::position state = chess::position::from_fen(chess::position::fen_start);
    mcts::Network network;
    std::vector<SelfPlayWorker::GameRow> game_rows;
    size_t moves = 0;


    while(!state.is_checkmate() && !state.is_stalemate() && moves++ < max_moves) {
        std::shared_ptr<mcts::Node> root = mcts::mcts(state, max_iter, network);
        GameRow row = {state, root->action_distribution(num_actions)};
        game_rows.push_back(row);
        std::cout << state.to_string() << std::endl;
        chess::move move = root->best_move();
        print_row(row);
        state.make_move(move);
    }
    std::cout << "Self play game done. Rows: " << game_rows.size() << std::endl;
}

void SelfPlayWorker::print_row(const SelfPlayWorker::GameRow& row) {
    size_t n = row.action_distribution.size();
    std::vector<std::pair<double, size_t>> action_values(n);
    for (size_t i = 0; i < n; i++) {
        action_values[i] = std::make_pair(row.action_distribution[i], i);
    }
    std::sort(action_values.begin(), action_values.end());
    size_t num_items = 3;
    std::cout << "Best moves: ";
    for (size_t i = 1; i < num_items+1; i++) {
        auto p = action_values[n-i];
        double value = p.first;
        size_t action = p.second;
        chess::move move = mcts::Network::move_from_action(row.state, action);
        std::cout << "(" << move.to_lan() << ", " << value << ") - ";
    }
    std::cout << std::endl;
}
