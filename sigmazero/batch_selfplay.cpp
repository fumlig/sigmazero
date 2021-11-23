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
	sigmanet model(0, 64, 13);

	torch::Device device(torch::kCPU);
	// For now only use CPU for self-play (overhead for single evaluation on GPU):

    if(torch::cuda::is_available())
    {
         device = torch::Device(torch::kCUDA);
         std::cerr << "selfplay: Using CUDA" << std::endl;
    }
    else
	{
         std::cerr << "selfplay: Using CPU" << std::endl;
    }

	torch::load(model, model_path);
	model->to(device);
	model->eval();
	model->zero_grad();

	std::cerr << "loaded model" << std::endl;

	auto model_changed = std::filesystem::last_write_time(model_path);

	std::cerr << "started with model " << model_path << std::endl;

	// Sätt till 1.0 för att stänga av fast playouts
	double full_search_prob = 0.25;
	
	int full_search_iterations = 600;
	int fast_search_iterations = 100;

	std::bernoulli_distribution search_type_dist(full_search_prob);

	int model_check_counter = 0;
	int model_update_int = 100;

	// with 600 iterations:
	// 1 worker: ~1 second
	// 8 workers: ~4 seconds 
	// 32 workers: ~15 seconds
	// 64 workers: ~31 seconds
	// 256 workers: ~128 seconds
	int batch_size = 4;
	
	std::vector<selfplay_worker> workers(batch_size);

	while (true)
	{
		// Load every 100 iterations
		if(++model_check_counter == model_update_int)
		{
			model_check_counter = 0;
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
		}

		std::vector<chess::position> positions_to_evaluate(batch_size);
		std::vector<bool> position_mask(batch_size);
		
		// Initial stuff
		for(int worker_idx = 0 ; worker_idx < batch_size ; ++worker_idx)
		{
			positions_to_evaluate[worker_idx] = workers[worker_idx].get_position();
		}
		
		// Evaluate
		auto evaluation = model->evaluate_batch(positions_to_evaluate, device);
		for(int worker_idx = 0 ; worker_idx < batch_size ; ++worker_idx) 
		{
			workers[worker_idx].initial_setup(evaluation[worker_idx]);
		}
	
		// Do tha search
		bool do_full_search = search_type_dist(get_generator());
		int iters = do_full_search ? full_search_iterations : fast_search_iterations;

		for(int i = 0 ; i < iters ; ++i)
		{
		// Do stuff with workers
			for(int worker_idx = 0 ; worker_idx < batch_size ; ++worker_idx)
			{
				std::optional<chess::position> traversed_position = workers[worker_idx].traverse();
				if(!traversed_position)
				{
					// maybe remove? or not
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

		// Make the best moves
		for(int worker_idx = 0 ; worker_idx < batch_size ; ++worker_idx)
		{
			chess::move move = workers[worker_idx].make_best_move(model->encode_input(workers[worker_idx].get_position()), true);
			std::cerr << "Worker " << worker_idx << " made move " << move.to_lan() << std::endl;
			
			// Output game and reset worker
			if(workers[worker_idx].game_is_terminal()) 
			{
				workers[worker_idx].output_game(std::cout);
				workers[worker_idx] = selfplay_worker();
			}
		} 
	}
	return 0;
}
