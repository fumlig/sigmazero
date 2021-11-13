#include "mcts.hpp"
#include "node.hpp"
#include "misc.hpp"
#include <chess/chess.hpp>
#include <memory>
#include <sigmazero/drl/sigmanet.hpp>


namespace mcts
{

    std::shared_ptr<Node> mcts(chess::position state, int max_iter, sigmanet& network)
    {

        std::shared_ptr<Node> main_node{std::make_shared<Node>(state)};
        std::pair<double, std::unordered_map<size_t, double>> evaluation = network.evaluate(state);
        main_node->explore_and_set_priors(evaluation);
        main_node->add_exploration_noise(0.3, 0.25);
        for(int i = 0 ; i < max_iter ; ++i)
        {
            std::shared_ptr<Node> current_node = main_node->traverse();
            if(!current_node->is_over()) {
                current_node->backpropagate(current_node->get_terminal_value());
                continue;
            }
            evaluation = network.evaluate(current_node->get_state());
            current_node->explore_and_set_priors(evaluation);
        }
        return main_node;
    }
}
