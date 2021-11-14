#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>
#include <memory>
#include <sstream>
#include <random>
#include <algorithm>
#include <cstdint>
#include <utility>
#include <future>
#include <thread>

#include <chess/chess.hpp>
#include <torch/torch.h>

#include "dummynet.hpp"
#include "base64.hpp"


static torch::Tensor decode(const std::string& data)
{
    torch::Tensor tensor;
	std::istringstream stream(base64_decode(data));
	torch::load(tensor, stream);
	return tensor;
}


static std::string readline(std::istream& in)
{
	std::string line;
	std::getline(in, line);
	return line;
}


int main(int argc, char** argv)
{
	dummynet model(10, 20);

	if(argc != 2)
	{
		std::cerr << "missing model path" << std::endl;
		return 1;
	}

	std::filesystem::path model_path = argv[1];

	// save initial model
	torch::save(model, model_path);

	std::cerr << "saved initial model" << std::endl;
	std::cout << std::endl; // indicate that model has been updated

	// start training
	std::cerr << "starting training" << std::endl;

	const std::size_t window_size = 1024;
	const std::size_t batch_size = 256;

	std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> window;
	window.reserve(window_size);

	// rng for batch sampling
	std::random_device random_device;
    std::mt19937 generator(random_device());

	std::future<std::string> replay_future = std::async(readline, std::ref(std::cin));

	// statistics
	unsigned long long received = 0;
	unsigned long long consumed = 0;

	while(true)
	{
		while(replay_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
		{
			std::string replay_encoding = replay_future.get();
			std::cerr << "replay bytes: " << replay_encoding.size() << std::endl;

			std::string encoded_images, encoded_values, encoded_policies;
			std::istringstream(replay_encoding) >> encoded_images >> encoded_values >> encoded_policies;
			window.emplace_back(decode(encoded_images), decode(encoded_values), decode(encoded_policies));

			std::cerr << "replay received of size " << std::get<0>(window.back()).size(0) << std::endl;
			replay_future = std::async(readline, std::ref(std::cin));

			received++;
		}

		// erase replays outside window
		if(window.size() > window_size)
		{
			window.erase(window.begin(), window.end() - window_size);
		}

		// wait for enough games to be available
		if(window.size() < batch_size)
		{
			continue;
		}

		std::cout << "sampling batch" << std::endl;

		// sample batch of replays
		std::vector<torch::Tensor> sample_images;
		std::vector<torch::Tensor> sample_values;
		std::vector<torch::Tensor> sample_policies;

		sample_images.reserve(batch_size);
		sample_values.reserve(batch_size);
		sample_policies.reserve(batch_size);

		std::vector<std::size_t> weights;
		weights.reserve(window_size);
		for(const auto& replay: window) weights.push_back(std::get<0>(replay).size(0));
		std::discrete_distribution<std::size_t> replay_distribution(weights.begin(), weights.end());

		for(int i = 0; i < batch_size; i++)
		{
			std::size_t replay = replay_distribution(generator);
			std::uniform_int_distribution<std::size_t> index_distribution(0, weights.at(replay)-1);
			std::size_t index = index_distribution(generator);

			//std::cerr << "sampling replay " << replay << ", index " << index << std::endl;

    		using namespace torch::indexing;

			sample_images.push_back(std::get<0>(window.at(replay)).index({static_cast<int>(index)}));
			sample_values.push_back(std::get<1>(window.at(replay)).index({static_cast<int>(index)}));
			sample_policies.push_back(std::get<2>(window.at(replay)).index({static_cast<int>(index)}));

			consumed++;
		}

		torch::Tensor batch_images = torch::stack(sample_images);
		torch::Tensor batch_values = torch::stack(sample_values);
		torch::Tensor batch_policies = torch::stack(sample_policies);

		//std::cerr << "batch ready" << std::endl;

		// train on batch
		// todo
		std::this_thread::sleep_for(std::chrono::milliseconds(100));

		// update model
		//torch::save(model, model_path);
		std::cout << std::endl; // indicate that model has updated

		// show statistics
		std::cerr << "received: " << received << ", consumed: " << consumed << std::endl;
	}

	return 0;
}
