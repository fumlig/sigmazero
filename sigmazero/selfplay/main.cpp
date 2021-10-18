#include <iostream>
#include <thread>
#include <chrono>
#include <chess/chess.hpp>


int main(int argc, char** argv)
{
	std::cout << "running selfplay..." << std::endl;

	while(true)
	{
		std::cout << "selfplay replay" << std::endl;
		std::this_thread::sleep_for(std::chrono::seconds(1));
	}

	return 0;
}