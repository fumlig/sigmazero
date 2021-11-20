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
#include <functional>
#include <vector>

#include <chess/chess.hpp>
#include <torch/torch.h>

#include "drl/sigmanet.hpp"
#include "sync_queue.hpp"
#include "base64.hpp"


static torch::Tensor decode(const std::string& data)
{
    torch::Tensor tensor;
	std::istringstream stream(base64_decode(data));
	torch::load(tensor, stream);
	return tensor;
}


static void receive_replays(std::vector<std::ifstream>& files, sync_queue<std::string>& queue)
{
	std::vector<std::reference_wrapper<std::istream>> streams(files.begin(), files.end());

	if(files.empty())
	{
		streams.push_back(std::cin);
	}

	std::cerr << "started replay thread with " << streams.size() << " sources" << std::endl;

	while(true)
	{
		for(std::istream& stream: streams)
		{
			std::string replay;
			std::getline(stream, replay);
			queue.push(replay);
		}
	}
}


int main(int argc, char** argv)
{
	if(argc < 2)
	{
		std::cerr << "missing model path" << std::endl;
		return 1;
	}

	// setup initial model
	std::filesystem::path model_path(argv[1]);
	sigmanet model(0, 64, 10);

	if(std::filesystem::exists(model_path))
	{
		torch::load(model, model_path);
		std::cerr << "loaded existing model" << std::endl;
	}
	else
	{
		torch::save(model, model_path);
		std::cerr << "saved initial model" << std::endl;
		//std::cout << std::endl; // indicate that model has been updated
	}
	
	// start receiving replays
	std::vector<std::ifstream> replay_files(argv+2, argv+argc);
	sync_queue<std::string> replay_queue;
	std::thread replay_thread(receive_replays, std::ref(replay_files), std::ref(replay_queue));

	// check cuda support
	torch::Device device(torch::kCPU);
    if(torch::cuda::is_available())
    {
        device = torch::Device(torch::kCUDA);
        std::cerr << "Using CUDA" << std::endl;
    }
    else
	{
        std::cerr << "Using CPU" << std::endl;
    }

	model->train();
	model->to(device);
	torch::optim::SGD optimizer(model->parameters(),
    torch::optim::SGDOptions(0.2).momentum(0.9).weight_decay(0.0001)); // varying lr


	// statistics
	unsigned long long received = 0;
	unsigned long long consumed = 0;

	// replay window
	const std::size_t window_size = 64;
	const std::size_t batch_size = 16;

	torch::Tensor window_images;
	torch::Tensor window_values;
	torch::Tensor window_policies;

	bool first_replay = true;

	// start training
	std::cerr << "starting training" << std::endl;

	while(true)
	{
		while(replay_queue.size())
		{
			std::string replay = replay_queue.pop();

			std::string encoded_image;
			std::string encoded_value;
			std::string encoded_policy;

			std::istringstream(replay) >> encoded_image >> encoded_value >> encoded_policy;

			torch::Tensor replay_image = decode(encoded_image).unsqueeze(0);
			torch::Tensor replay_value = decode(encoded_value).unsqueeze(0);
			torch::Tensor replay_policy = decode(encoded_policy).unsqueeze(0);
			
			//std::cerr << replay_policy << std::endl;
			
			if(first_replay)
			{
				first_replay = false;
				window_images = replay_image;
				window_values = replay_value;
				window_policies = replay_policy;
			}
			else
			{
				window_images = torch::cat({window_images, replay_image}, 0);
				window_values = torch::cat({window_values, replay_value}, 0);
				window_policies = torch::cat({window_policies, replay_policy}, 0);
			}

			std::cerr << "selfplay result received" << std::endl;
			received++;
		}

		// wait for enough games to be available
		if(received < window_size)
		{
			continue;
		}
		std::cerr << "training on window" << std::endl;
		// remove old replays
		torch::indexing::Slice window_slice(-window_size);

		window_images = window_images.index({window_slice});
		window_values = window_values.index({window_slice});
		window_policies = window_policies.index({window_slice});

		// sample batch of replays
		torch::Tensor batch_sample = torch::randint(window_size, {batch_size}).to(torch::kLong);

		torch::Tensor batch_images = window_images.index({batch_sample}).to(device);
		torch::Tensor batch_values = window_values.index({batch_sample}).to(device);
		torch::Tensor batch_policies = window_policies.index({batch_sample}).to(device);

		//std::cerr << "batch ready" << std::endl;
		// train on batch
		model->zero_grad();
		auto [value, policy] = model->forward(batch_images);
		//std::cerr << "distribution label: " << batch_policies << std::endl;
		auto loss = sigma_loss(value, batch_values, policy, batch_policies);
		loss.backward();
		optimizer.step();
		std::cerr << "loss: " << loss.item<float>() << std::endl;
		consumed += batch_size;

		// update model
		torch::save(model, model_path);
		//std::cout << std::endl; // indicate that model has updated

		// show statistics
		std::cerr << "received: " << received << ", consumed: " << consumed << std::endl;
	}

	return 0;
}
