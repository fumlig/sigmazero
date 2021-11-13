#include <sigmazero/drl/sigmanet.hpp>
#include <chess/chess.hpp>
#include <iostream>
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
    return 0;
}