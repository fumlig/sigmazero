#include <sigmazero/drl/sigmanet.hpp>
#include <sigmazero/mcts/node.hpp>
#include <chess/chess.hpp>
#include <iostream>
#include <memory>
/*
Perform mcts with an untrained sigmanet to verify that the program does not crash
*/
int main()
{
    chess::init();
    sigmanet network(1, 64, 10);
    chess::position start = chess::position::from_fen(chess::position::fen_start);
    // Perform the whole mcts shabang
    
    // Initial evaluation and expansion
    std::pair<double, std::unordered_map<size_t, double>> evaluation = network.evaluate(start);

    for(auto[action, value] : evaluation.second)
    {
        std::cout << "action: " << action << ", value: " << value << std::endl;
    }    

    std::cout << "position value: " << evaluation.first << std::endl;

    std::cout << "attempting one monte carlo tree search..." << std::endl;


    std::shared_ptr<mcts::Node> main_node{std::make_shared<mcts::Node>(start)};
    std::pair<double, std::unordered_map<size_t, double>> evaluation = network.evaluate(start);
    main_node->explore_and_set_priors(evaluation);
    main_node->add_exploration_noise(0.3, 0.25);
    for(int i = 0 ; i < 100 ; ++i)
    {
        std::shared_ptr<mcts::Node> current_node = main_node->traverse();
        if(!current_node->is_over()) {
            current_node->backpropagate(current_node->get_terminal_value());
            continue;
        }
        evaluation = network.evaluate(current_node->get_state());
        current_node->explore_and_set_priors(evaluation);
    }

    return 0;
}