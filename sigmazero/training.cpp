#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>
#include <chess/chess.hpp>


int main(int argc, char** argv)
{
	if(argc != 2)
	{
		std::cerr << "training:\tmissing model path" << std::endl;
		return 1;
	}
 
	std::filesystem::path model_path = argv[1];

	std::ofstream(model_path).close();

	std::cerr << "training:\tcreated initial model" << std::endl;

	std::cerr << "training:\tstarted with model " << model_path << std::endl;

	std::string replay;

	int batch_size = 10;
	int i = 0;

	while(std::getline(std::cin, replay))
	{
		std::cout << "training:\treplay received: " << replay << std::endl;
		
		if(++i == batch_size)
		{
			std::cout << "training:\tupdating model on batch" << std::endl;
			std::ofstream(model_path).close(); // touch model file
			i = 0;	
		}
	}

	return 0;
}