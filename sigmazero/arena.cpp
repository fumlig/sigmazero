#include <iostream>
#include <memory>

#include <chess/chess.hpp>

#include "search.hpp"
#include "sigmanet.hpp"
#include "rules.hpp"


int main(int argc, char** argv)
{
    const int games = 10;
    const int simulations = 2500;
    const auto value_function = material_value;

    if(argc < 3)
    {
        std::cerr << "missing model paths" << std::endl;
        return 1;
    }

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
	torch::NoGradGuard no_grad;

    sigmanet model_a = make_network();
    sigmanet model_b = make_network();

    torch::load(model_a, argv[1]);
    torch::load(model_b, argv[2]);

	model_a->to(device);
	model_a->eval();
	model_a->zero_grad();

	model_b->to(device);
	model_b->eval();
	model_b->zero_grad();

    float score_a = 0.0f;
    float score_b = 0.0f;

    chess::side seat = chess::side_white;

    chess::init();

    for(int i = 0; i < games; i++)
    {
        chess::game game;

        while(!game.is_terminal())
        {
            chess::side turn = game.get_position().get_turn();
            sigmanet model = turn == seat ? model_a : model_b;
            bool noise = game.size() == 0;
            std::shared_ptr<node> best = run_mcts(game, model, device, stop_after(simulations), noise);
            chess::move move = best->move;

            game.push(move);
            std::cout << move.to_lan() << " " << std::flush;
        }

        std::cout << *game.get_score(seat) << "-" << *game.get_score(chess::opponent(seat)) << " (" << value_function(game, seat) << ")" << std::endl;

        score_a += *game.get_score(seat);
        score_b += *game.get_score(chess::opponent(seat));

        seat = chess::opponent(seat);
    }

    std::cout << score_a << "-" << score_b << std::endl;

    return 0;
}