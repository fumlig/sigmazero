#include <iostream>
#include <thread>
#include <chrono>
#include <string>
#include <filesystem>
#include <random>
#include <memory>

#include <chess/chess.hpp>
#include <torch/torch.h>

#include "selfplay_worker.hpp"
#include "drl/sigmanet.hpp"
#include "mcts/node.hpp"
#include "base64.hpp"
#include "util.hpp"


int main(int argc, char **argv)
{
	if (argc != 2)
	{
		std::cerr << "missing model path" << std::endl;
		return 1;
	}
	else
	{
		std::cerr << "using model path" << argv[1] << std::endl; 
	}

	chess::init();
	torch::NoGradGuard no_grad;
	std::filesystem::path model_path(argv[1]);

	// wait for initial model
	while (!std::filesystem::exists(model_path))
	{
		std::cerr << "waiting for initial model" << std::endl;
		std::this_thread::sleep_for(std::chrono::seconds(1));
	}

	// load initial model
	sigmanet model(0, 128, 10);

	torch::Device device(torch::kCPU);
	// For now only use CPU for self-play (overhead for single evaluation on GPU):

    if(torch::cuda::is_available())
    {
         device = torch::Device(torch::kCUDA);
         std::cerr << "using CUDA" << std::endl;
    }
    else
	{
         std::cerr << "using CPU" << std::endl;
    }

	torch::load(model, model_path);
	model->to(device);
	model->eval();
	model->zero_grad();

	std::cerr << "loaded model" << std::endl;

	auto model_changed = std::filesystem::last_write_time(model_path);

	std::cerr << "started with model " << model_path << std::endl;

	// Sätt till 1.0 för att stänga av fast playouts
	double full_search_prob = 1.0;
	
	int full_search_iterations = 800;
	int fast_search_iterations = 100;

	std::bernoulli_distribution search_type_dist(full_search_prob);

	int batch_size = 256;
	
	std::vector<selfplay_worker> workers(batch_size);

	unsigned long long total_moves = 0;

	while (true)
	{
		// load latest model
		auto model_write = std::filesystem::last_write_time(model_path);
		if (model_write > model_changed)
		{
			try
			{
				torch::load(model, model_path);
				model->to(device);
				model_changed = model_write;
				std::cerr << "updated model loaded" << std::endl;
			}
			catch(const std::exception& e)
			{
				std::cerr << "loading updated model failed" << std::endl;
			}
		}
	
		std::cerr << "checked for new model" << std::endl;

		std::vector<chess::position> positions_to_evaluate(batch_size);
		std::vector<bool> position_mask(batch_size);
		
		// Initial stuff
		for(int worker_idx = 0 ; worker_idx < batch_size ; ++worker_idx)
		{
			positions_to_evaluate[worker_idx] = workers[worker_idx].get_position();
		}
		// Evaluate
		auto evaluation = model->evaluate_batch(positions_to_evaluate, device);
		
		std::cerr << "batch evaluated" << std::endl;

		for(int worker_idx = 0 ; worker_idx < batch_size ; ++worker_idx) 
		{
			workers[worker_idx].initial_setup(evaluation[worker_idx]);
		}

		std::cerr << "workers setup" << std::endl;
	
		// Do tha search
		bool do_full_search = search_type_dist(get_generator());
		int iters = do_full_search ? full_search_iterations : fast_search_iterations;

		std::cerr << "doing " << (do_full_search ? "full" : "fast") << " search" << std::endl;

		for(int i = 0 ; i < iters ; ++i)
		{
			// Do stuff with workers
			for(int worker_idx = 0 ; worker_idx < batch_size ; ++worker_idx)
			{
				std::optional<chess::position> traversed_position = workers[worker_idx].traverse();
				if(!traversed_position)
				{
					position_mask[worker_idx] = false;
					continue;
				}
				position_mask[worker_idx] = true;
				positions_to_evaluate[worker_idx] = *traversed_position;
			}

			auto evaluation = model->evaluate_batch(positions_to_evaluate, device);

			for(int worker_idx = 0 ; worker_idx < batch_size ; ++worker_idx)
			{
				if (position_mask[worker_idx]) workers[worker_idx].explore_and_set_priors(evaluation[worker_idx]);
			}
		}

		std::cerr << "mcts step done" << std::endl;

		int terminal_count = 0;

		// Make the best moves
		for(int worker_idx = 0 ; worker_idx < batch_size ; ++worker_idx)
		{
			selfplay_worker& worker = workers.at(worker_idx);

			chess::move move = worker.make_best_move(model->encode_input(worker.get_position()), do_full_search);
			//std::cerr << "worker " << worker_idx << ": " << move.to_lan() << std::endl;
			
			// Output game and reset worker
			if(worker.game_is_terminal()) 
			{
				if(worker.replay_size() == 0)
				{
					std::cerr << "worker " << worker_idx << " terminal with replay size " << 0 << ", skipping send" << std::endl;
				}
				else
				{
					std::cerr << "worker " << worker_idx << " sending replay of size " << worker.replay_size() << " with " << workers[worker_idx].get_game().size() << " moves: ";
					for(const auto& [move, _]: workers[worker_idx].get_game().get_history())
					{
						std::cerr << move.to_lan() << " ";
					}
					std::cerr << std::endl;
					workers[worker_idx].output_game(std::cout);
				}

				workers[worker_idx] = selfplay_worker();
				terminal_count++;
			}
		}

		total_moves += batch_size;

		std::cerr << "batch complete, " << terminal_count << " workers reset, " << total_moves << " total moves" << std::endl;
	}
	return 0;
}
