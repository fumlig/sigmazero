#include <iostream>
#include <thread>
#include <chrono>
#include <string>
#include <filesystem>
#include <chess/chess.hpp>


int main(int argc, char** argv)
{
	if(argc != 2)
	{
		std::cerr << "selfplay:\tmissing model path" << std::endl;
		return 1;
	}

	std::filesystem::path model_path(argv[1]);

	if(!std::filesystem::exists(model_path) || !std::filesystem::is_regular_file(model_path))
	{
		std::cerr << "selfplay:\tmodel does not exist or is not a file" << std::endl;
		return 1;
	}

	auto model_changed = std::filesystem::last_write_time(model_path);

	std::cerr << "selfplay:\tstarted with model " << model_path << std::endl;

	while(true)
	{
		auto model_write = std::filesystem::last_write_time(model_path);

		if(model_write > model_changed)
		{
			std::cerr << "selfplay:\tnew model found" << std::endl;
			model_changed = model_write;
		}

		std::cout << "<game moves> <score> <policy>" << std::endl;
		std::this_thread::sleep_for(std::chrono::seconds(1));
	}

	return 0;
}