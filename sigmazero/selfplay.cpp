#include <iostream>
#include <thread>
#include <chrono>
#include <string>
#include <filesystem>

#include <chess/chess.hpp>
#include <torch/torch.h>
#include <torch/script.h>


int main(int argc, char** argv)
{
	if(argc != 2)
	{
		std::cerr << "missing model path" << std::endl;
		return 1;
	}

	std::filesystem::path model_path(argv[1]);

	while(!std::filesystem::exists(model_path))
	{
		std::cerr << "model does not exist, waiting..." << std::endl;
		std::this_thread::sleep_for(std::chrono::seconds(1));
	}

	torch::jit::script::Module model;

	model = torch::jit::load(model_path);
	std::cerr << "loaded model" << std::endl;

	auto model_changed = std::filesystem::last_write_time(model_path);

	std::cerr << "started with model " << model_path << std::endl;

	while(true)
	{
		auto model_write = std::filesystem::last_write_time(model_path);

		if(model_write > model_changed)
		{
			std::cerr << "updated model loaded" << std::endl;
			model = torch::jit::load(model_path);
			model_changed = model_write;
		}

		std::cerr << "sending selfplay game" << std::endl;

		std::cout << "<game moves> <score> <policy>" << std::endl;
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
	}

	return 0;
}
