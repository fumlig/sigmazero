#include <iostream>
#include <thread>
#include <chrono>
#include <string>
#include <filesystem>
#include <sstream>
#include <random>
#include <memory>
#include <cstdint>
#include <functional>

#include <chess/chess.hpp>
#include <torch/torch.h>

#include "rules.hpp"
#include "sigmanet.hpp"
#include "search.hpp"
#include "base64.hpp"
#include "utility.hpp"


static std::string encode(const torch::Tensor &tensor)
{
	std::ostringstream data;
	torch::save(tensor, data);
	return base64_encode(data.str());
}

struct worker
{
	std::shared_ptr<node> root;
	chess::game game = chess::game(chess::position::from_fen("1k1r4/pp4p1/1n4p1/2p3Pp/2P3n1/1P3NP1/P4PB1/1K2R3 b - - 0 31"), {}); // Kasparov vs. Deep Blue
	//chess::game game = chess::game(chess::position::from_fen("ppppk3/ppppppp1/ppppppp1/ppppppp1/8/8/PPPPPPPN/PPPPKPPR w K - 0 1"), {});

	chess::game scratch_game;
	std::vector<std::shared_ptr<node>> search_path;

	std::vector<torch::Tensor> images;
	std::vector<torch::Tensor> visits;
	std::vector<torch::Tensor> values;
	std::vector<chess::side> turns;

	torch::Tensor make_image()
	{
		return game_image(game);
	}

	void expand_root(const torch::Tensor policy)
	{
		root = std::make_shared<node>();
		root->expand(game, policy);
		add_exploration_noise(*root);
	}

	torch::Tensor traverse_tree()
	{
		std::tie(search_path, scratch_game) = traverse(root, game);
		return game_image(scratch_game);
	}

	void expand_leaf(torch::Tensor policy)
	{
		std::shared_ptr<node> leaf = search_path.back();
		leaf->expand(scratch_game, policy);
	}

	void backpropagate_path(torch::Tensor value, bool use_terminal_value = true)
	{
		std::optional<int> v = scratch_game.get_value(chess::opponent(scratch_game.get_position().get_turn()));
		
		if(use_terminal_value && v)
		{
			value = torch::tensor(static_cast<float>(*v));
		}

		backpropagate(search_path, value, scratch_game.get_position().get_turn());
	}

	void save_image(std::function<float(const chess::game&, chess::side)> value_function)
	{
		images.push_back(game_image(game));
		visits.push_back(root->child_visits());
		values.push_back(torch::tensor(value_function(game, chess::opponent(game.get_position().get_turn())))); // seems like we have to use opponent here, some mistake in mcts?
		turns.push_back(game.get_position().get_turn());
	}

	void make_move()
	{
		std::shared_ptr<node> best = root->select_best();
		game.push(best->move);
		root = best;
	}

	bool is_terminal(std::size_t max_moves = 512)
	{
		return game.is_terminal() || game.size() >= max_moves;
	}

	void send_replay(std::ostream& out, bool use_terminal_value = true)
	{
		std::size_t replay_size = images.size();

		if(replay_size == 0)
		{
			return;
		}

		for(std::size_t i = 0; i < replay_size; i++)
		{
			torch::Tensor image = images[i];
			torch::Tensor value = values[i];
			torch::Tensor policy = visits[i];

			if(use_terminal_value)
			{
				std::optional<int> v = game.get_value(chess::opponent(turns[i])); // todo: maybe this should actually be inverted?
				value = torch::tensor(v ? static_cast<float>(*v) : 0.0f);
			}

			std::cout << encode(image) << ' ' << encode(value) << ' ' << encode(policy) << std::endl;
		}

		images.clear();
		visits.clear();
		values.clear();
		turns.clear();
	}
};


int main(int argc, char **argv)
{
	// configuration
	const int full_search_iterations = 800;
	const int fast_search_iterations = 100;

	const float fast_search_prob = 0.0f;

	const int max_moves = 512;
	const int batch_size = 64;

	const bool send_on_termination = true;

	const std::function<float(const chess::game&, chess::side)> value_function = material_value;

	// statistics

	long searches = 0;
	long sent = 0;
	int white_wins = 0;
	int black_wins = 0;
	int draws = 0;

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

	if(torch::cuda::is_available())
	{
		std::cerr << "cuda available" << std::endl;
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
	bool fill_window = false;

	std::vector<worker> workers(batch_size);
	std::vector<torch::Tensor> batch_images(batch_size);

	while(true)
	{
		// load latest model
		auto model_write = std::filesystem::last_write_time(model_path);
		if(model_write > model_changed)
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

		auto [_, batch_policies] = model->forward(torch::stack(batch_images).to(device));

		// expand roots		
		for(int i = 0; i < batch_size; i++)
		{
			workers[i].expand_root(batch_policies.index({i}));
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
				workers[i].expand_leaf(batch_policies.index({i}));
				workers[i].backpropagate_path(batch_values.index({i}));
			}
		}

		for(int i = 0; i < batch_size; i++)
		{
			// hope that filling the initial window with replays of poor quality is ok...
			if(!fast_search || fill_window)
			{
				workers[i].save_image(value_function);
			}

			workers[i].make_move();

			bool send_replay = !send_on_termination || workers[i].is_terminal(max_moves);

			if(send_replay)
			{
				sent += workers[i].images.size();
				workers[i].send_replay(std::cout, send_on_termination);
			}

			if(workers[i].is_terminal(max_moves))
			{
				std::cerr << "terminal: " << workers[i].game.size() << " plies, ";
				
				for(auto [move, _]: workers[i].game.get_history())
				{
					std::cerr << move.to_lan() << " ";
				}

				std::optional<int> value = workers[i].game.get_value();
				std::string outcome = "-";

				if(value)
				{
					white_wins += *value > 0;
					black_wins += *value < 0;
					draws += *value == 0;

					outcome = std::to_string(*value);
				}

				std::cerr << outcome << std::endl;

				workers[i] = worker{};
			}
		}

		searches++;

		std::cerr << "batch: " << searches << " searches, " << searches*batch_size << " moves, " << sent << " sent, " << white_wins << " white wins, " << black_wins << " black wins, " << draws << " draws" << std::endl; 
	}

	return 0;
}
