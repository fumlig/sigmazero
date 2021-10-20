#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>
#include <memory>

#include <chess/chess.hpp>
#include <torch/torch.h>

#include "dummynet.hpp"


std::ostream& info()
{
	return std::cerr << "training:\t";
}


int main(int argc, char** argv)
{
	if(argc != 2)
	{
		info() << "missing model path" << std::endl;
		return 1;
	}
 
	std::filesystem::path model_path = argv[1];
	dummynet model;

	if(std::filesystem::exists(model_path))
	{
		info() << "loading existing model" << std::endl;
		torch::load(model, model_path);
	}
	else
	{
		info() << "saving initial model" << std::endl;
		torch::save(model, model_path);
	}

	std::cout << "<new model event>" << std::endl;

	info() << "starting training" << std::endl;

	std::string replay;
	int batch_size = 100;
	int i = 0;

	while(std::getline(std::cin, replay))
	{
		info() << "replay received: " << replay << std::endl;
		
		if(++i == batch_size)
		{
			info() << "updating model on batch" << std::endl;
			i = 0;

			info() << "saving updated model" << std::endl;

			torch::save(model, model_path);
			std::cout << "<new model event>" << std::endl;
		}
	}

	return 0;
}