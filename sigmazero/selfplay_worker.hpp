#ifndef SIGMAZERO_SELFPLAY_WORKER_HPP
#define SIGMAZERO_SELFPLAY_WORKER_HPP

#include <chess/chess.hpp>
#include <torch/torch.h>

#include "drl/sigmanet.hpp"
#include "mcts/node.hpp"

#include <utility>
#include <optional>
#include <unordered_map>
#include <ostream>
#include <cstdint>


class selfplay_worker
{
public:
    selfplay_worker();
    const chess::position& get_position() const;
    const chess::game& get_game() const;
    bool game_is_terminal(size_t max_game_size=512) const;
    std::size_t replay_size() const;
    void initial_setup(const std::pair<double, std::unordered_map<size_t, double>>& evaluation);

    std::optional<chess::position> traverse();
    void explore_and_set_priors(const std::pair<double, std::unordered_map<size_t, double>>& evaluation);
    chess::move make_best_move(torch::Tensor position_encoding, const bool& record);
    void output_game(std::ostream& stream);

private:

    static std::string encode(const torch::Tensor &tensor);
    chess::game game{};
    std::vector<torch::Tensor> images{};
    std::vector<torch::Tensor> policies{};
    std::vector<chess::side> players{};
    std::shared_ptr<mcts::Node> main_node;
    std::shared_ptr<mcts::Node> current_node;
};



#endif