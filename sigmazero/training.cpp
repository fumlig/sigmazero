#include <iostream>
#include <string>
#include <chess/chess.hpp>


int main(int argc, char** argv)
{
	std::string replay;

	while(std::getline(std::cin, replay))
	{
		std::cout << "training replay: " << replay << std::endl;
	}

	return 0;
}