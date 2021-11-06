#include <random>
#include <vector>
#include <atomic>
#include <chrono>

#include <chess/chess.hpp>
#include <uci/uci.hpp>
#include <mcts/node.hpp>
#include <mcts/rollout.hpp>
#include <mcts/policy.hpp>
#include <mcts/policy_handcrafted.hpp>
#include "random_mcts_engine.hpp"

random_mcts_engine::random_mcts_engine():
root(),
random{},
generator{random()},
policy{generator},
rollout{50} // 50 rollout sims
{
	// most clients require these options
	opt.add<uci::option_spin>("MultiPV", 1, 1, 1);
	opt.add<uci::option_spin>("Move Overhead", 0, 0, 1);
	opt.add<uci::option_spin>("Threads", 1, 1, 1);
	opt.add<uci::option_spin>("Hash", 1, 1, 1);

	// demo options
	opt.add<uci::option_check>("Demo Check", true);
	opt.add<uci::option_spin>("Demo Spin", 0, -10, 10);
	opt.add<uci::option_combo>("Demo Combo", "Apples", std::initializer_list<std::string>{"Apples", "Oranges", "Bananas"});
	opt.add<uci::option_string>("Demo String", "Tjenare");
	opt.add<uci::option_button>("Demo Button", []()
	{
		std::cerr << "button pressed" << std::endl;
	});
}


void random_mcts_engine::setup(const chess::position& position, const std::vector<chess::move>& moves)
{
	root = position;
	for(const chess::move& move: moves)
	{
		root.make_move(move);
	}
}

uci::search_result random_mcts_engine::search(const uci::search_limit& limit, uci::search_info& info, const std::atomic_bool& ponder, const std::atomic_bool& stop)
{
	info.message("search started");

	chess::side side = root.get_turn();
	float max_time = std::min(limit.time, 10.0f); // estimate ~100 moves per game
	auto start_time = std::chrono::steady_clock::now();
	int current_depth = 1;

	std::shared_ptr<node::Node> main_node{std::make_shared<node::Node>(root, side)};
	main_node->expand();

	while(!stop)
	{
		std::shared_ptr<node::Node> current_node = main_node->traverse();

		if(current_node->is_over()) break;
		if(current_node->get_n() != 0)
		{
			current_node->expand();
			current_node = current_node->get_children().front();
		}
		current_node->rollout(rollout, policy);
		current_node->backpropagate();
		
		info.depth(8008);
		info.nodes(1337);

		bool early_stop = !limit.infinite && !ponder;
		auto current_time = std::chrono::steady_clock::now();
		std::chrono::duration<double> elapsed_time = current_time - start_time;

		if(stop || (early_stop && elapsed_time.count() >= max_time)) break;

	}
	//info.line(...);
	return {main_node->best_move(), std::nullopt};
	
}

void random_mcts_engine::reset()
{
	generator.seed(generator.default_seed);
}
