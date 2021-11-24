#include <iostream>
#include <thread>
#include <chrono>
#include <string>
#include <filesystem>
#include <sstream>
#include <random>
#include <memory>

#include <chess/chess.hpp>
#include <torch/torch.h>

#include "drl/sigmanet.hpp"
#include "mcts/node.hpp"
#include "base64.hpp"
#include "util.hpp"

static std::string encode(const torch::Tensor &tensor)
{
	std::ostringstream data;
	torch::save(tensor, data);
	return base64_encode(data.str());
}

int main(int argc, char **argv)
{
	if (argc != 2)
	{
		std::cerr << "missing model path" << std::endl;
		return 1;
	}
	else
	{
		std::cerr << "using model path " << argv[1] << std::endl
	}

	chess::init();
	torch::NoGradGuard no_grad;
	std::filesystem::path model_path(argv[1]);

	// wait for initial model
	while (!std::filesystem::exists(model_path))
	{
		std::cerr << "waiting for initial model" << std::endl;
		std::this_thread::sleep_for(std::chrono::seconds(1));
	}

	// load initial model
	sigmanet model(0, 128, 10);

	torch::Device device(torch::kCPU);
	// For now only use CPU for self-play (overhead for single evaluation on GPU):

    // if(torch::cuda::is_available())
    // {
    //     device = torch::Device(torch::kCUDA);
    //     std::cerr << "Using CUDA" << std::endl;
    // }
    // else
	// {
    //     std::cerr << "Using CPU" << std::endl;
    // }

	torch::load(model, model_path);
	model->to(device);
	model->eval();
	model->zero_grad();

	std::cerr << "loaded model" << std::endl;

	auto model_changed = std::filesystem::last_write_time(model_path);

	std::cerr << "started with model " << model_path << std::endl;

	// Sätt till 1.0 för att stänga av fast playouts
	double full_search_prob = 0.25;
	
	int full_search_iterations = 600;
	int fast_search_iterations = 100;

	std::bernoulli_distribution search_type_dist(full_search_prob);

	while (true)
	{
		// load latest model
		auto model_write = std::filesystem::last_write_time(model_path);
		if (model_write > model_changed)
		{
			try
			{
				torch::load(model, model_path);
				model->to(device);
				model_changed = model_write;
				std::cerr << "updated model loaded" << std::endl;
			}
			catch(const std::exception& e)
			{
				std::cerr << "loading updated model failed" << std::endl;
			}
		}

		// start new game
		std::cerr << "new game started" << std::endl;
		chess::game game;
		std::vector<torch::Tensor> images{};
		std::vector<torch::Tensor> policies{};
		std::vector<chess::side> players{};

		std::shared_ptr<mcts::Node> main_node{std::make_shared<mcts::Node>(game.get_position())};
		while (!game.is_terminal() && game.size() <= 100) // TODO: Check end
		{
			auto evaluation = model->evaluate(game.get_position(), device);
			main_node->explore_and_set_priors(evaluation);
			main_node->add_exploration_noise(0.3, 0.25);

			bool do_full_search = search_type_dist(get_generator());
			int iters = do_full_search ? full_search_iterations : fast_search_iterations;

			for (int i = 0; i < iters; ++i)
			{
				std::shared_ptr<mcts::Node> current_node = main_node->traverse();
				if (current_node->is_over())
				{
					current_node->backpropagate(current_node->get_terminal_value()); //TODO: terminal value according to mate
					continue;
				}
				evaluation = model->evaluate(current_node->get_state(), device);
				current_node->explore_and_set_priors(evaluation);
			}

			// Save move as training data if full search was done
			if (do_full_search) {
				
				// todo: extract from position
				images.push_back(model->encode_input(game.get_position()));
				std::vector<double> action_dist = main_node->action_distribution();
				torch::Tensor action_tensor = torch::tensor(action_dist);
				policies.push_back(action_tensor);
				players.push_back(game.get_position().get_turn());
			}


			//std::cerr << "action dist " << torch::tensor(main_node->action_distribution()) << std::endl;
			// next position
			main_node = main_node->best_child();
			main_node->make_start_node();
			chess::move best_move = main_node->get_move();
			std::cerr << "making move " << best_move.to_lan() << std::endl;
			
			game.push(best_move);
		}

		// send tensors
		for(size_t i = 0 ; i < images.size(); ++i)
		{
			std::optional<int> game_value = game.get_value(players[i]);
			float terminal_value = game_value ? *game_value : 0.0f;
			torch::Tensor value = torch::tensor(terminal_value);

			std::cout << encode(images[i]) << ' ' << encode(value) << ' ' << encode(policies[i]) << std::endl; //according to side
		}

		std::cerr << "sent replay of size " << images.size() << std::endl;
	}

	return 0;
}
