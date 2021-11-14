#include <iostream>
#include <thread>
#include <chrono>
#include <string>
#include <filesystem>
#include <sstream>
#include <random>

#include <chess/chess.hpp>
#include <torch/torch.h>

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

	std::filesystem::path model_path(argv[1]);

	// wait for initial model
	while(!std::filesystem::exists(model_path))
	{
		std::cerr << "model does not exist, waiting..." << std::endl;
		std::this_thread::sleep_for(std::chrono::seconds(1));
	}

	// load initial model
	dummynet model(10, 20);
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
			std::cerr << "updated model loaded" << std::endl;
			torch::load(model, model_path);
			model_changed = model_write;
		}

		// start new game
		chess::game game;

		// replay data
		std::vector<torch::Tensor> images;
		std::vector<torch::Tensor> values;
		std::vector<torch::Tensor> policies;

		while(!game.is_checkmate() && game.size() <= 50)
		{
			// todo: extract from position
			torch::Tensor image = torch::zeros(10);
			torch::Tensor value = torch::zeros(1);
			torch::Tensor policy = torch::zeros(20);

			// save in replay
			images.push_back(image);
			values.push_back(value);
			policies.push_back(policy);

			// next position
			game.push(game.get_position().moves().front());
		}

		// send replay
		//std::cerr << "sending selfplay game" << std::endl;

		torch::Tensor replay_images = torch::stack(images);
		torch::Tensor replay_values = torch::stack(values);
		torch::Tensor replay_policies = torch::stack(policies);

		std::cout << encode(replay_images) << ' ' << encode(replay_values) << ' ' << encode(replay_policies) << std::endl;

		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}

	return 0;
}
