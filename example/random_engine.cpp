#include <random>
#include <vector>
#include <atomic>
#include <chrono>

#include <chess/chess.hpp>
#include <uci/uci.hpp>

#include "random_engine.hpp"



random_engine::random_engine():
root(),
random{},
generator{random()}
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


void random_engine::setup(const chess::position& position, const std::vector<chess::move>& moves)
{
	root = position;

	for(const chess::move& move: moves)
	{
		root.make_move(move);
	}
}

uci::search_result random_engine::search(const uci::search_limit& limit, uci::search_info& info, const std::atomic_bool& ponder, const std::atomic_bool& stop)
{
	info.message("search started");

	chess::side side = root.get_turn();
	std::vector<chess::move> moves = root.moves();
	std::vector<chess::move> best = {moves[0]};
	float max_time = std::min(limit.time, limit.clocks[side] / 100); // estimate ~100 moves per game
	auto start_time = std::chrono::steady_clock::now();
	int current_depth = 1;

	while(!stop)
	{
		info.depth(current_depth);
		info.nodes(std::pow(moves.size(), current_depth)); // overestimate

		// search moves
		for(std::size_t i = 0; i < moves.size(); i++)
		{
			const chess::move& move = moves[i];
			info.move(move, i);

			// stop if time is up
			bool early_stop = !limit.infinite && !ponder;
			auto current_time = std::chrono::steady_clock::now();
			std::chrono::duration<double> elapsed_time = current_time - start_time;

			if(stop || (early_stop && elapsed_time.count() >= max_time))
			{
				goto done; // innan du säger något: håll käften
			}

			// update best line to random line of depth
			best.resize(current_depth);
			chess::position p = root;

			for(int j = 0; j < current_depth; j++)
			{
				// choose random move
				std::vector<chess::move> ms = p.moves();
				std::uniform_int_distribution<int> distribution(0, ms.size()-1);
				chess::move m = ms.at(distribution(generator));
				best[j] = m;
				p = p.copy_move(m);

				// was mate accidentally found?
				if(p.is_checkmate())
				{
					info.mate(p.fullmove() - root.fullmove());

					if(early_stop)
					{
						goto done;
					}
				}
			}

			info.line(best);

			// fake search time
			std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(std::pow(2, current_depth))));
		}

		// next depth reached...
		current_depth++;
	}

done: // se ovan
	return {best.front(), std::nullopt};
}

void random_engine::reset()
{
	generator.seed(generator.default_seed);
}
