#include "mcts.hpp"
#include "node.hpp"
#include "misc.hpp"
#include "network.hpp"
#include <chess/chess.hpp>
#include <memory>

namespace mcts
{

    std::shared_ptr<Node> mcts(chess::position state, int max_iter, const mcts::Network& network)
    {

        std::shared_ptr<Node> main_node{std::make_shared<Node>(state)};
        main_node->explore_and_set_priors(network);
        main_node->add_exploration_noise(0.3, 0.25);
        for(int i = 0 ; i < max_iter ; ++i)
        {
            std::shared_ptr<Node> current_node = main_node->traverse();
            if(!current_node->is_over()) {
                current_node->backpropagate(current_node->get_terminal_value());
                continue;
            }
            current_node->explore_and_set_priors(network);
        }
        return main_node;
    }
}
