#include <iostream>
#include <memory>
#include <unordered_map>
#include <string>

#include <torch/torch.h>
#include <chess/chess.hpp>
#include <uci/uci.hpp>

#include "random_engine.hpp"


int main(int argc, char** argv)
{
	if(torch::cuda::is_available())
	{
		std::cout << "cuda available" << std::endl;
	}
	else
	{
		std::cout << "cuda not available" << std::endl;
	}

	torch::Tensor tensor = torch::rand({3,3});
  	std::cerr << tensor << std::endl;
	std::cerr << torch::flipud(tensor) << std::endl;

	chess::init();
	random_engine engine;
	return uci::main(engine);
}
