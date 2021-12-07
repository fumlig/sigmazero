#include <limits>

#include "search.hpp"
#include "utility.hpp"
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


void node::expand(const chess::game& game, const torch::Tensor policy)
{
    turn = game.get_position().get_turn();

    const std::vector<chess::move>& legal_moves = game.get_moves();
    
    float policy_sum = 0.0f;
    torch::Tensor policy_exp = torch::exp(policy);

    for(chess::move move: legal_moves)
    {
        policy_sum += policy_exp.index({move_action(move, game)}).item<float>();
    }

    for(chess::move move: legal_moves)
    {
        std::shared_ptr<node> child = std::make_shared<node>();

        child->action = move_action(move, game);
        child->move = move;
        child->turn = chess::opponent(turn);
        child->prior = policy_exp.index({child->action}).item<float>()/policy_sum;

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

std::shared_ptr<node> node::select_best() const
{
    int max_visit_count = -std::numeric_limits<int>::infinity();
    std::shared_ptr<node> best;

    // todo: alphazero has softmax_sample for short games
    for(std::shared_ptr<node> child: children)
    {
        if(child->visit_count > max_visit_count)
        {
            best = child;
            max_visit_count = child->visit_count;
        }
    }

    return best;
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

std::pair<std::vector<std::shared_ptr<node>>, chess::game> traverse(std::shared_ptr<node> root, const chess::game& game)
{
    std::shared_ptr<node> leaf = root;
    chess::game scratch_game = game;
    std::vector<std::shared_ptr<node>> search_path = {leaf};

    while(leaf->expanded())
    {
        leaf = leaf->select_child();
        scratch_game.push(leaf->move);
        search_path.push_back(leaf);
    }

    return {search_path, scratch_game};
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



stop_after::stop_after(int limit): simulation{0}, limit{limit}
{

}

bool stop_after::operator()(const node&)
{
    return ++simulation >= limit;
}



std::shared_ptr<node> run_mcts(const chess::game& game, sigmanet network, torch::Device device, stop_cond stop, bool noise, std::optional<std::shared_ptr<node>> last_best)
{
    auto root = std::make_shared<node>();

    if(last_best)
    {
        root = *last_best;
    }

    auto image = game_image(game);
    auto [value, policy] = network->forward(image.unsqueeze(0).to(device));

    if(!root->expanded())
    {
        root->expand(game, policy.squeeze());
    }

    if(noise)
    {
        add_exploration_noise(*root);
    }

    while(!stop(*root))
    {
        auto [search_path, scratch_game] = traverse(root, game);
        auto leaf = search_path.back();
                
        std::optional<int> v = scratch_game.get_value(chess::opponent(scratch_game.get_position().get_turn()));

		if(v)
		{
			value = torch::tensor(static_cast<float>(*v));
            backpropagate(search_path, value, scratch_game.get_position().get_turn());
		}
        else
        {
            image = game_image(scratch_game);
            auto [value, policy] = network->forward(image.unsqueeze(0).to(device));
            leaf->expand(scratch_game, policy.squeeze());
            backpropagate(search_path, value.squeeze(), scratch_game.get_position().get_turn());
        }

    }

    return root->select_best();
}
