#include <iostream>
#include <memory>
#include <unordered_map>
#include <string>
#include <mcts/node.hpp>
#include <chess/chess.hpp>
#include <uci/uci.hpp>

#include "random_mcts_engine.hpp"


int main(int argc, char** argv)
{
	node::init(1, 0, 2.0);
	chess::init();
	random_mcts_engine engine;
	return uci::main(engine);
}
