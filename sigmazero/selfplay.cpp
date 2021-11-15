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
	sigmanet model(0, 64, 10);
	torch::load(model, model_path);
	model->eval();
	model->zero_grad();
	std::default_random_engine generator;
	std::cerr << "loaded model" << std::endl;

	auto model_changed = std::filesystem::last_write_time(model_path);

	std::cerr << "started with model " << model_path << std::endl;

	while (true)
	{
		// load latest model
		auto model_write = std::filesystem::last_write_time(model_path);
		if (model_write > model_changed)
		{
			torch::load(model, model_path);
			model_changed = model_write;
			std::cerr << "updated model loaded" << std::endl;
		}

		// start new game
		std::cerr << "new game started" << std::endl;
		chess::game game;
		std::vector<torch::Tensor> images{};
		std::vector<torch::Tensor> policies{};

		while (!game.is_checkmate() && !game.is_stalemate() && game.size() <= 50) // TODO: Check end
		{
			std::shared_ptr<mcts::Node> main_node{std::make_shared<mcts::Node>(game.get_position())};
			auto evaluation = model->evaluate(game.get_position());
			main_node->explore_and_set_priors(evaluation);
			main_node->add_exploration_noise(0.3, 0.25, generator);

			for (int i = 0; i < 80; ++i)
			{
				std::shared_ptr<mcts::Node> current_node = main_node->traverse();
				if (current_node->is_over())
				{
					current_node->backpropagate(current_node->get_terminal_value()); //TODO: terminal value according to mate
					continue;
				}
				evaluation = model->evaluate(current_node->get_state());
				current_node->explore_and_set_priors(evaluation);
			}
			// todo: extract from position
			images.push_back(model->encode_input(game.get_position()));
			policies.push_back(torch::tensor(main_node->action_distribution()));
			
			//std::cerr << "action dist " << torch::tensor(main_node->action_distribution()) << std::endl;
			// next position
			chess::move best_move = main_node->best_move();
			//std::cerr << "Making move " << best_move.to_lan() << std::endl;
			game.push(best_move);
		}

		// send tensors
		float terminal_value;
		if(game.is_checkmate())
		{
			terminal_value = game.get_position().get_turn() == chess::side_white ? -1 : 1;
		}
		else
		{
			terminal_value = 0;
		}
		for(int i = 0 ; i < game.size() ; ++i) {
			torch::Tensor value = torch::tensor(i%2 == 0 ? terminal_value : -terminal_value);
			std::cout << encode(images[i]) << ' ' << encode(value) << ' ' << encode(policies[i]) << std::endl; //according to side
		}
	}

	return 0;
}
