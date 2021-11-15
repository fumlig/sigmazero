#include <algorithm>
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
#include <sigmazero/mcts/node.hpp>
#include <sigmazero/drl/sigmanet.hpp>

#include "dummynet.hpp"
#include "base64.hpp"


static std::string encode(const torch::Tensor& tensor)
{
    std::ostringstream data;
    torch::save(tensor, data);
    return base64_encode(data.str());
}


int main(int argc, char** argv)
{
	if(argc != 2)
	{
		std::cerr << "missing model path" << std::endl;
		return 1;
	}

	chess::init();
	std::default_random_engine generator; // To add exploration noise
	std::filesystem::path model_path(argv[1]);

	// wait for initial model
	while(!std::filesystem::exists(model_path))
	{
		std::cerr << "waiting for initial model" << std::endl;
		std::this_thread::sleep_for(std::chrono::seconds(1));
	}

	// load initial model
	sigmanet model(0, 64, 50);
	torch::load(model, model_path);
	
	std::cerr << "loaded model" << std::endl;

	auto model_changed = std::filesystem::last_write_time(model_path);

	std::cerr << "started with model " << model_path << std::endl;

	while(true)
	{
		// load latest model
		auto model_write = std::filesystem::last_write_time(model_path);
		if(model_write > model_changed)
		{
			torch::load(model, model_path);
			model_changed = model_write;
			std::cerr << "updated model loaded" << std::endl;
		}

		// start new game
		std::cerr << "new game started" << std::endl;
		chess::game game;

		// replay data
		std::vector<torch::Tensor> images;
		std::vector<torch::Tensor> values;
		std::vector<torch::Tensor> policies;

		int mcts_iter = 80;
		while(!game.is_checkmate() && game.size() <= 50)
		{
			std::shared_ptr<mcts::Node> main_node{std::make_shared<mcts::Node>(game.get_position())};
			std::pair<double, std::unordered_map<size_t, double>> evaluation = model.evaluate(game.get_position());
			main_node->explore_and_set_priors(evaluation);
			main_node->add_exploration_noise(0.3, 0.25, generator);

			for(int i = 0 ; i < mcts_iter ; ++i) {
				std::shared_ptr<mcts::Node> current_node = main_node->traverse();
				if(current_node->is_over()) {
					current_node->backpropagate(current_node->get_terminal_value());
					continue;
				}
				std::cout << "iteration i=" << i << ", evaluating...";
				evaluation = model.evaluate(current_node->get_state());
				current_node->explore_and_set_priors(evaluation);
			}
			// todo: extract from position
			torch::Tensor image = model.encode_input(game.get_position());
			torch::Tensor value = torch::zeros(1); //???
			std::vector<double> action_distribution = main_node->action_distribution();
			torch::Tensor policy = torch::from_blob((double*)action_distribution.data(), action_distribution.size());

			// save in replay
			images.push_back(image); // TODO: Flip board if necessary etc.
			values.push_back(value);
			policies.push_back(policy);
			// send tensors
			std::cout << encode(image) << ' ' << encode(value) << ' ' << encode(policy) << std::endl;

			// next position
			game.push(main_node->best_move());
		}

		// Fill value with final score
		//values.reserve(images.size());
		//double final_value = 0; // TODO: assign correct value
		//std::fill(values.begin(), values.end(), torch::tensor(final_value));

		// send replay
		//std::cerr << "sending selfplay game" << std::endl;

		torch::Tensor replay_images = torch::stack(images);
		torch::Tensor replay_values = torch::stack(values);
		torch::Tensor replay_policies = torch::stack(policies);

		std::cout << encode(replay_images) << ' ' << encode(replay_values) << ' ' << encode(replay_policies) << std::endl;

		// todo: remove this, just here to avoid congestion of communication channel
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}

	return 0;
}
