#include <iostream>
#include <thread>
#include <chrono>
#include <string>
#include <filesystem>

#include <chess/chess.hpp>
#include <torch/torch.h>

#include "dummynet.hpp"

std::ostream& info()
{
	return std::cerr << "selfplay:\t";
}


int main(int argc, char** argv)
{
	if(argc != 2)
	{
		info() << "missing model path" << std::endl;
		return 1;
	}

	std::filesystem::path model_path(argv[1]);
	
	while(!std::filesystem::exists(model_path))
	{
		info() << "model does not exist, waiting..." << std::endl;
		std::this_thread::sleep_for(std::chrono::seconds(1));
	} 

	dummynet model;
	torch::load(model, model_path);

	info() << "loaded model" << std::endl;

	auto model_changed = std::filesystem::last_write_time(model_path);

	info() << "started with model " << model_path << std::endl;

	while(true)
	{
		auto model_write = std::filesystem::last_write_time(model_path);

		if(model_write > model_changed)
		{
			info() << "updated model loaded" << std::endl;
			torch::load(model, model_path);
			model_changed = model_write;
		}

		std::cout << "<game moves> <score> <policy>" << std::endl;
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
	}
 
	return 0;
}