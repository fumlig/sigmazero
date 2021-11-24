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
#include <queue>
#include <iomanip>

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


static void replay_receiver(std::istream& stream, sync_queue<std::string>& queue)
{
	while(true)
	{
		std::string replay;
		std::getline(stream, replay);
		queue.push(replay);
	}
}


int main(int argc, char** argv)
{
	if(argc < 2)
	{
		std::cerr << "missing model path" << std::endl;
		return 1;
	}
	else
	{
		std::cerr << "using model path " << argv[1] << std::endl;
	}

	// setup initial model
	std::filesystem::path model_path(argv[1]);
	sigmanet model(0, 64, 13);

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
	
	// receive selfplay replays
	std::vector<std::ifstream> replay_files(argv+2, argv+argc);
	sync_queue<std::string> replay_queue;
	std::vector<std::reference_wrapper<std::istream>> replay_streams(replay_files.begin(), replay_files.end());
	std::vector<std::thread> replay_threads;
	std::queue<std::chrono::time_point<std::chrono::steady_clock>> replay_timestamps;

	if(replay_streams.empty())
	{
		// fall back to stdin
		replay_streams.push_back(std::cin);
	}

	std::cerr << "reading replays from " << replay_streams.size() << " streams" << std::endl;

	for(std::istream& replay_stream: replay_streams)
	{
		// one thread per stream is ok since they will mostly be blocked
		replay_threads.emplace_back(replay_receiver, std::ref(replay_stream), std::ref(replay_queue));
	}

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
    torch::optim::SGDOptions(0.01).momentum(0.9).weight_decay(0.0001)); // varying lr


	// statistics
	unsigned long long received = 0;
	unsigned long long consumed = 0;

	// replay window
	const std::size_t window_size = 256;
	const std::size_t batch_size = 64;

	torch::Tensor window_images;
	torch::Tensor window_values;
	torch::Tensor window_policies;

	const unsigned save_rate = 16;			// save after this number of batches
	const unsigned checkpoint_rate = 256;	// checkpoint after this number of saves

	unsigned batches_since_save = 0;
	unsigned saves_since_checkpoint = 0;

	bool first_replay = true;

	// start training
	std::cerr << "starting training" << std::endl;

	while(true)
	{
		while(replay_queue.size())
		{
			std::string replay = replay_queue.pop();
			replay_timestamps.push(std::chrono::steady_clock::now());

			std::string encoded_image;
			std::string encoded_value;
			std::string encoded_policy;

			std::istringstream(replay) >> encoded_image >> encoded_value >> encoded_policy;

			torch::Tensor replay_image;
			torch::Tensor replay_value;
			torch::Tensor replay_policy;

			try
			{
				replay_image = decode(encoded_image).unsqueeze(0);
				replay_value = decode(encoded_value).unsqueeze(0);
				replay_policy = decode(encoded_policy).unsqueeze(0);
			}
			catch(const std::exception& e)
			{
				std::cerr << "exception raised when decoding replay tensors" << std::endl;
				std::cerr << e.what() << '\n';
				continue;
			}
			
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
			received++;
		}

		// wait for enough games to be available
		if(received < window_size)
		{
			continue;
		}
		// remove old replays
		torch::indexing::Slice window_slice(-window_size);

		window_images = window_images.index({window_slice});
		window_values = window_values.index({window_slice});
		window_policies = window_policies.index({window_slice});

		// remove old timestamps
		while(replay_timestamps.size() > window_size)
		{
			replay_timestamps.pop();
		}

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
		if(++batches_since_save == save_rate)
		{
			batches_since_save = 0;
			
			torch::save(model, model_path);
			std::cerr << "saved model " << model_path << std::endl;

			if(++saves_since_checkpoint == checkpoint_rate)
			{
				saves_since_checkpoint = 0;

				auto now = std::chrono::system_clock::now();
 				const std::time_t t_c = std::chrono::system_clock::to_time_t(now);
				std::ostringstream out;
				out << "ckpt_" << std::put_time(std::localtime(&t_c), "%FT%T") << ".pt";
				
				std::filesystem::path checkpoint_path = out.str();
				std::filesystem::copy_file(model_path, checkpoint_path);
				
				std::cerr << "saved checkpoint " << checkpoint_path << std::endl;
			}
		}

		//std::cout << std::endl; // indicate that model has updated

		// show statistics
		std::chrono::duration<float> window_duration = replay_timestamps.back() - replay_timestamps.front();
		std::cerr << "received: " << received << ", consumed: " << consumed << ", rate: " << window_size/window_duration.count() << std::endl;
	}

	return 0;
}
