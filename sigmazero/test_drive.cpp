#include <sigmazero/drl/sigmanet.hpp>
#include <sigmazero/mcts/node.hpp>
#include <chess/chess.hpp>
#include <iostream>
#include <memory>
/*
Perform mcts with an untrained sigmanet to verify that the program does not crash
*/
void print_evaluation(std::pair<double, std::unordered_map<size_t, double>> evaluation)
{
    for(auto[action, value] : evaluation.second)
    {
        std::cout << "action: " << action << ", value: " << value << std::endl;
    }    

    std::cout << "position value: " << evaluation.first << std::endl;
}

int main()
{
    chess::init();
    sigmanet network(1, 64, 10);
    chess::position start = chess::position::from_fen(chess::position::fen_start);
    // Perform the whole mcts shabang
    
    // Initial evaluation and expansion
    std::pair<double, std::unordered_map<size_t, double>> evaluation = network.evaluate(start);
    print_evaluation(evaluation);
    std::cout << "attempting one monte carlo tree search with 1000 iterations..." << std::endl;
    std::default_random_engine generator;

    std::shared_ptr<mcts::Node> main_node{std::make_shared<mcts::Node>(start)};
    evaluation = network.evaluate(start);
    main_node->explore_and_set_priors(evaluation);

    main_node->add_exploration_noise(0.3, 0.25, generator);

    for(int i = 0 ; i < 1000 ; ++i)
    {
        std::cout << "traversing...";
        std::shared_ptr<mcts::Node> current_node = main_node->traverse();
        std::cout << "traversed. Is over?";
        if(current_node->is_over()) {
            std::cout << "Yes. Backpropagating:";
            current_node->backpropagate(current_node->get_terminal_value());
            std::cout << "Done. Next iter:" << std::endl;
            continue;
        }
        std::cout << "iteration i=" << i << ", evaluating...";
        evaluation = network.evaluate(current_node->get_state());
        std::cout << "evaluated. Explore and set priors:";
        current_node->explore_and_set_priors(evaluation);
        std::cout << "Done. Next iter:" << std::endl;
    }

    std::cout << "search done. Evaluation of current position:" << std::endl;
    print_evaluation(evaluation);
    std::cout << "estimated best move: " << main_node->best_move().to_lan() << std::endl;


    return 0;
}