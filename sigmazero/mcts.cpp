#include <limits>

#include "mcts.hpp"
#include "util.hpp"
#include "sigmanet.hpp"
#include "rules.hpp"


bool node::expanded() const
{
    return children.size() > 0;
}

float node::value() const
{
    if(visit_count == 0)
    {
        return 0;
    }
    
    return value_sum / visit_count;
}


void node::expand(const chess::game& game, const torch::Tensor value, const torch::Tensor policy)
{
    turn = game.get_position().get_turn();

    if(game.get_position().is_terminal())
    {
        return;
    }

    std::vector<chess::move> legal_moves = game.get_position().moves();
    
    float policy_sum = 0.0f;
    torch::Tensor policy_exp = torch::exp(policy);

    for(chess::move move: legal_moves)
    {
        int action = move_action(move, turn);
        policy_sum += policy_exp.index({action}).item<float>();
    }

    for(chess::move move: legal_moves)
    {
        std::shared_ptr<node> child = std::make_shared<node>();
        int action = move_action(move, turn);

        child->prior = policy_exp.index({action}).item<float>()/policy_sum;
        child->move = move;
        child->action = action;
        child->turn = chess::opponent(turn);

        children.push_back(child);
    }
}


std::shared_ptr<node> node::select_child() const
{
    float max_score = -std::numeric_limits<float>::infinity();
    std::shared_ptr<node> selected;
    
    for(std::shared_ptr<node> child: children)
    {
        float score = ucb_score(*this, *child);

        if(score > max_score)
        {
            selected = child;
            max_score = score;
        }
    }

    return selected;
}

chess::move node::select_move() const
{
    int max_visit_count = -std::numeric_limits<int>::infinity();
    chess::move move;

    // todo: alphazero has softmax_sample for short games
    for(std::shared_ptr<node> child: children)
    {
        if(child->visit_count > max_visit_count)
        {
            move = child->move;
            max_visit_count = child->visit_count;
        }
    }

    return move;
}

torch::Tensor node::child_visits() const
{
    torch::Tensor visits = torch::zeros({num_actions});
    int sum_visits = 0;

    for(std::shared_ptr<node> child: children)
    {
        sum_visits += child->visit_count;
    }

    for(std::shared_ptr<node> child: children)
    {
        using namespace torch::indexing;
        visits.index_put_({child->action}, static_cast<float>(child->visit_count) / sum_visits);
    }

    return visits;
}


float ucb_score(const node& parent, const node& child, float pb_c_base, float pb_c_init)
{
    float pb_c = std::log((parent.visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init;
    pb_c *= std::sqrt(parent.visit_count) / (child.visit_count + 1);

    float prior_score = pb_c * child.prior;
    float value_score = child.value();

    return prior_score + value_score;
}


void add_exploration_noise(node& root, float dirichlet_alpha, float exploration_fraction)
{
    std::mt19937& rand_engine = get_generator();
    std::gamma_distribution<float> gamma_dist(dirichlet_alpha, 1.0f);

    for(std::shared_ptr<node> child: root.children)
    {
        float noise = gamma_dist(rand_engine);
        child->prior = child->prior*(1 - exploration_fraction) + noise*exploration_fraction;
    }
}

std::pair<chess::game, std::vector<std::shared_ptr<node>>> traverse(std::shared_ptr<node> root, const chess::game& game)
{
    std::shared_ptr<node> leaf = root;
    chess::game scratch_game = game;
    std::vector<std::shared_ptr<node>> search_path = {leaf};

    while(leaf->expanded())
    {
        leaf = leaf->select_child();
        scratch_game.push(action_move(leaf->action, chess::opponent(leaf->turn)));
        search_path.push_back(leaf);
    }

    return {scratch_game, search_path};
}

void backpropagate(std::vector<std::shared_ptr<node>>& search_path, const torch::Tensor value, chess::side turn)
{
    for(std::shared_ptr<node> node: search_path)
    {
        //node->value_sum += node->turn == turn ? value.item<float>() : (1.0f - value.item<float>());
        node->value_sum += node->turn == turn ? value.item<float>() : -value.item<float>();
        node->visit_count += 1;
    }
}


chess::move run_mcts(const chess::game& game, sigmanet network, int simulations)
{
    return {};

#if 0
    std::shared_ptr<node> root = std::make_shared<node>();
    evaluate(*root, game, network);
    add_exploration_noise(*root);

    for(int i = 0; i < simulations; i++)
    {
        std::shared_ptr<node> leaf = root;
        std::vector<std::shared_ptr<node>> search_path = {leaf};
        chess::game scratch_game = game;

        while(leaf->expanded())
        {
            leaf = leaf->select_child();
            scratch_game.push(leaf->move);
            search_path.push_back(leaf);
        }

        torch::Tensor value = evaluate(*leaf, scratch_game, network);
        backpropagate(search_path, value, scratch_game.get_position().get_turn());
    }

    return root->select_move();
#endif
}