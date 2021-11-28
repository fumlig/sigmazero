#include <iostream>
#include <thread>
#include <chrono>
#include <string>
#include <filesystem>
#include <sstream>
#include <random>
#include <memory>
#include <cstdint>

#include <chess/chess.hpp>
#include <torch/torch.h>

#include "rules.hpp"
#include "sigmanet.hpp"
#include "mcts.hpp"
#include "base64.hpp"
#include "util.hpp"


static std::string encode(const torch::Tensor &tensor)
{
	std::ostringstream data;
	torch::save(tensor, data);
	return base64_encode(data.str());
}


struct worker
{
	chess::game game;
	std::shared_ptr<node> root;

	chess::game scratch_game;
	std::vector<std::shared_ptr<node>> search_path;

	std::vector<torch::Tensor> images;
	std::vector<torch::Tensor> visits;
	std::vector<chess::side> turns;

	worker():
	game(),
	root{},
	scratch_game(),
	search_path(),
	images(),
	visits(),
	turns()
	{
	}

	~worker() = default;

	torch::Tensor make_image()
	{
		return game_image(game);
	}

	void expand_root(const torch::Tensor value, const torch::Tensor policy)
	{
		root = std::make_shared<node>();
		root->expand(game, value, policy);
		add_exploration_noise(*root);
	}

	torch::Tensor traverse_tree()
	{
		std::tie(scratch_game, search_path) = traverse(root, game);
		return game_image(scratch_game);
	}

	void expand_leaf(const torch::Tensor value, const torch::Tensor policy)
	{
		if(scratch_game.get_position().is_terminal())
		{
			return;
		}

		std::shared_ptr<node> leaf = search_path.back();
		leaf->expand(scratch_game, value, policy);
	}

	void backpropagate_path(const torch::Tensor value)
	{
		backpropagate(search_path, value, scratch_game.get_position().get_turn());
	}

	void save_image()
	{
		images.push_back(game_image(game));
		visits.push_back(root->child_visits());
		turns.push_back(game.get_position().get_turn());
	}

	void make_move()
	{
		chess::move move = root->select_move();
		game.push(move);
	}

	bool is_terminal(std::size_t max_moves = 512)
	{
		return game.get_position().is_terminal() || game.size() >= max_moves;
	}

	void send_replay(std::ostream& out)
	{
		std::size_t replay_size = images.size();

		if(replay_size == 0)
		{
			return;
		}

		for(std::size_t i = 0; i < replay_size; i++)
		{
			chess::side turn = turns[i];
			std::optional<int> game_value = game.get_value(turn);
        	float terminal_value = game_value ? *game_value : 0.0f;
			torch::Tensor value = torch::tensor(terminal_value);

			std::cout << encode(images[i]) << ' ' << encode(value) << ' ' << encode(visits[i]) << std::endl; //according to side
		}
	}
};


int main(int argc, char **argv)
{
	// configuration
	const int full_search_iterations = 800;
	const int fast_search_iterations = 100;

	const float fast_search_prob = 0.75;

	const int max_moves = 512;
	const int batch_size = 256;

	// statistics

	long searches = 0;
	
	long win_terminations = 0;
	long draw_terminations = 0;
	long early_terminations = 0;

	long white_wins = 0;
	long black_wins = 0;

	if(argc != 2)
	{
		std::cerr << "missing model path" << std::endl;
		return 1;
	}
	else
	{
		std::cerr << "using model path " << argv[1] << std::endl;
	}

	chess::init();
	torch::NoGradGuard no_grad;
	std::filesystem::path model_path(argv[1]);

	// wait for initial model
	while(!std::filesystem::exists(model_path))
	{
		std::cerr << "waiting for initial model" << std::endl;
		std::this_thread::sleep_for(std::chrono::seconds(1));
	}

	// load initial model
	sigmanet model = make_network();
	torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

	torch::load(model, model_path);
	model->to(device);
	model->eval();
	model->zero_grad();

	std::cerr << "loaded model" << std::endl;

	auto model_changed = std::filesystem::last_write_time(model_path);
	std::bernoulli_distribution search_type_dist(fast_search_prob);
	bool fill_window = true;

	std::vector<worker> workers(batch_size);
	std::vector<torch::Tensor> batch_images(batch_size);

	while(true)
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

			// training has started and window is full
			fill_window = false;
		}

		// initial evaluation
		for(int i = 0; i < batch_size; i++)
		{
			batch_images[i] = workers[i].make_image();
		}

		auto [batch_values, batch_policies] = model->forward(torch::stack(batch_images).to(device));

		// expand roots		
		for(int i = 0; i < batch_size; i++)
		{
			workers[i].expand_root(batch_values.index({i}), batch_policies.index({i}));
		}

		// tree search
		int simulations = full_search_iterations;
		bool fast_search = search_type_dist(get_generator());

		if(fast_search || fill_window)
		{
			simulations = fast_search_iterations;
		}

		for(int simulation = 0; simulation < simulations; simulation++)
		{
			for(int i = 0; i < batch_size; i++)
			{
				batch_images[i] = workers[i].traverse_tree();
			}

			auto [batch_values, batch_policies] = model->forward(torch::stack(batch_images).to(device));

			for(int i = 0; i < batch_size; i++)
			{
				workers[i].expand_leaf(batch_values.index({i}), batch_policies.index({i}));
				workers[i].backpropagate_path(batch_values.index({i}));
			}
		}

		for(int i = 0; i < batch_size; i++)
		{
			// hope that filling the initial window with replays of poor quality is ok...
			if(!fast_search || fill_window)
			{
				workers[i].save_image();
			}

			workers[i].make_move();

			if(workers[i].is_terminal(max_moves))
			{
				std::optional<float> value = workers[i].game.get_value();
				if(value)
				{
					white_wins += *value == 1;
					black_wins += *value == -1;

					draw_terminations += *value == 0;
					win_terminations += *value != 0;
				}
				else
				{
					early_terminations++;
				}

				std::cerr << "replay: " << workers[i].images.size() << " images, " << workers[i].game.size() << " plies, ";
				for(auto [move, _]: workers[i].game.get_history())
				{
					std::cerr << move.to_lan() << " ";
				}
				std::cerr << std::endl;

				workers[i].send_replay(std::cout);
				workers[i] = worker();
			}
		}

		searches++;
		
		std::cerr << "batch: " << searches << " searches, " << searches*batch_size << " moves, " << white_wins << " white wins, " << black_wins << " black wins, " << draw_terminations << " draws, " << early_terminations << " early terminations" << std::endl; 
	}

	return 0;
}
