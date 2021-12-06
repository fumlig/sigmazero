#ifndef SEARCH_HPP
#define SEARCH_HPP


#include <cmath>
#include <memory>
#include <vector>
#include <functional>
#include <optional>

#include <chess/chess.hpp>
#include <torch/torch.h>

#include "sigmanet.hpp"


struct node
{
    float prior{0.0f};
    chess::side turn{chess::side_none};    
    
    int action{-1};
    chess::move move{};

    int visit_count{0};
    float value_sum{0.0f};

    std::vector<std::shared_ptr<node>> children;

    bool expanded() const;
    float value() const;

    void expand(const chess::game& game, const torch::Tensor policy);

    std::shared_ptr<node> select_best() const;
    std::shared_ptr<node> select_child() const;
    torch::Tensor child_visits() const;
};


float ucb_score(const node& parent, const node& child, float pb_c_base = 19652, float pb_c_init = 1.25);
void add_exploration_noise(node& root, float dirichlet_alpha = 0.3f, float exploration_fraction = 0.25f);

std::pair<std::vector<std::shared_ptr<node>>, chess::game> traverse(std::shared_ptr<node> root, const chess::game& game);
void backpropagate(std::vector<std::shared_ptr<node>>& search_path, const torch::Tensor value, chess::side turn);


struct counter
{
    int count;
    const int limit;

    counter(int limit);
    bool operator()();
};

std::shared_ptr<node> run_mcts(const chess::game& game, sigmanet network, torch::Device device, std::function<bool()> stop, bool noise = false, std::optional<std::shared_ptr<node>> last_best = std::nullopt);


#endif