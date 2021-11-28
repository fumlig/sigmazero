#include <iostream>
#include <thread>
#include <chrono>
#include <string>
#include <filesystem>
#include <random>
#include <memory>

#include <chess/chess.hpp>
#include <torch/torch.h>

#include "../selfplay_worker.hpp"
#include "../drl/sigmanet.hpp"
#include "../mcts/node.hpp"
#include "../base64.hpp"
#include "../util.hpp"

int main() 
{
    // Generate selfplay batch
    chess::init();

    sigmanet model(0, 128, 10);

	torch::Device device(torch::kCPU);
	// For now only use CPU for self-play (overhead for single evaluation on GPU):

    if(torch::cuda::is_available())
    {
         device = torch::Device(torch::kCUDA);
         std::cerr << "using CUDA" << std::endl;
    }
    else
	{
         std::cerr << "using CPU" << std::endl;
    }
    model->to(device);
    model->eval();
	model->zero_grad();

    int batch_size = 32; // Amount of moves to train on
    selfplay_worker worker;
    std::vector<chess::position> positions_to_evaluate(1);
    bool position_mask = true;
    for(int m = 0 ; m < batch_size ; ++m)
    {
        positions_to_evaluate[0] = worker.get_position();
        auto evaluation = model->evaluate_batch(positions_to_evaluate, device);
        worker.initial_setup(evaluation[0]);

        for(int i = 0 ; i < 80 ; ++i)
        {
            std::optional<chess::position> traversed_position = worker.traverse();
            if(!traversed_position)
            {
                position_mask = false;
                std::cerr << "walked to terminal node" << std::endl;
                break;
            }
            position_mask = true;
            positions_to_evaluate[0] = *traversed_position;
            evaluation = model->evaluate_batch(positions_to_evaluate, device);
            if(position_mask) worker.explore_and_set_priors(evaluation[0]);
        }

        chess::move made_move = worker.make_best_move(model->encode_input(worker.get_position()), true);
        std::cerr << "amount of moves: " << worker.get_game().size() << std::endl;
    }
    worker.output_game(std::cout);
}