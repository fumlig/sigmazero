#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>
#include <memory>

#include <chess/chess.hpp>
#include <torch/torch.h>

#include "dummynet.hpp"


int main(int argc, char** argv)
{
	if(argc != 2)
	{
		std::cerr << "missing model path" << std::endl;
		return 1;
	}

	std::filesystem::path model_path = argv[1];
	dummynet model(10, 20);

	torch::save(model, model_path);

	std::cerr << "saved initial model" << std::endl;

	std::cout << std::endl; // indicate that model has been updated

	std::cerr << "starting training" << std::endl;

	std::string replay;
	int batch_size = 25;
	int i = 0;

	while(std::getline(std::cin, replay))
	{
		std::cerr << "replay received: " << replay << std::endl;

		if(++i == batch_size)
		{
			std::cerr << "updating model on batch" << std::endl;
			i = 0;

			std::cerr << "saving updated model" << std::endl;

			torch::save(model, model_path);
			std::cout << std::endl; // indicate that model has updated
		}
	}

	return 0;
}
