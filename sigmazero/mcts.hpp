#ifndef MCTS_HPP
#define MCTS_HPP


#include <cmath>
#include <memory>
#include <vector>

#include <chess/chess.hpp>
#include <torch/torch.h>

#include "sigmanet.hpp"


struct node
{
    float prior = 0.0f;
    chess::side turn{chess::side_none};
    
    chess::move move;
    int action;

    int visit_count = 0;
    float value_sum = 0.0f;

    std::vector<std::shared_ptr<node>> children;

    bool expanded() const;
    float value() const;

    void expand(const chess::game& game, const torch::Tensor value, const torch::Tensor policy);

    chess::move select_move() const;
    std::shared_ptr<node> select_child() const;
    torch::Tensor child_visits() const;
};


float ucb_score(const node& parent, const node& child, float pb_c_base = 19652, float pb_c_init = 1.25);
void add_exploration_noise(node& root, float dirichlet_alpha = 0.3f, float exploration_fraction = 0.25f);

std::pair<chess::game, std::vector<std::shared_ptr<node>>> traverse(std::shared_ptr<node> root, const chess::game& game);
void backpropagate(std::vector<std::shared_ptr<node>>& search_path, const torch::Tensor value, chess::side turn);

chess::move run_mcts(const chess::game& game, sigmanet network, int simulations);


#endif