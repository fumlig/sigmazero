#include <iostream>
#include <chess/chess.hpp>
#include "sigmanet.hpp"

void save_model(sigmanet& net, const std::string& path) {
    torch::serialize::OutputArchive output;
    net.save(output);
    output.save_to(path);
}

/*
Initializes sigmanet model
Trains the model using dummy data
Performs inference and asserts that output looks correct
*/
int main()
{
    std::cout << "Testing sigmanet model..." << std::endl;
    torch::Device device(torch::kCPU);
    // Check cuda support
    if(torch::cuda::is_available())
    {
        device = torch::Device(torch::kCUDA);
        std::cout << "Using CUDA" << std::endl;
    }
    else {
        std::cout << "Using CPU" << std::endl;
    }

    int filters = 64;
    int n_blocks = 10;
    double c = 0.0001; // L2 Regularization
    // Create dummy input data
    long batch_size = 64;
    int history = 3;
    int n_moves = 8 * 8 * 73; // Change in sigmanet as well

    std::cout << "Initializing model with parameters" <<
    "#filters " << filters << std::endl <<
    "#residual blocks " << n_blocks << std::endl <<
    "history " << history << std::endl;

    // Initialize model loss and optimizer
    sigmanet model(history, filters, n_blocks);
    std::cout << "initialised model" << std::endl;

    torch::Tensor input_state = torch::rand({batch_size, model.get_input_channels(), 8, 8}, device);
    // Create dummy output data
    torch::Tensor value_targets = torch::rand({batch_size, 1}, device);
    torch::Tensor policy_targets = torch::rand({batch_size, n_moves}, device);

    std::cout << "randomised data" << std::endl;
    

    model.to(device);
    model.train();

    // TODO: Move model to gpu if available?

    std::cout << "Training" << std::endl;
    torch::optim::SGD optimizer(model.parameters(), 
    torch::optim::SGDOptions(0.01).momentum(0.9).weight_decay(c));
    auto loss_fn = sigma_loss; // Defined in model.h

    // Training!
    int n_epochs = 2;

    for(int i = 0 ; i < n_epochs ; i++)
    {

        std::cout << "Epoch " << i << std::endl;

        model.zero_grad();
        auto[value, policy] = model.forward(input_state);
        auto loss = loss_fn(value, value_targets, policy, policy_targets);
        std::cout << "epoch " << i << "loss: " << loss.item<float>() << std::endl;
        loss.backward();
        optimizer.step();
    }
    std::cout << "Testing" << std::endl;
    // Inference
    model.eval();

    chess::init();
    
    chess::game game;
    game.push(game.get_position().moves().front());
    torch::Tensor test_state = model.encode_input(game.get_position()).to(device).unsqueeze(0);
    auto[value, policy] = model.forward(test_state);

    std::cout << "input encoding:" << std::endl;
    std::cout << model.encode_input(game.get_position()) << std::endl;

    std::cout << "inference result: " << std::endl << "value: " << value << std::endl << "policy: " << policy << std::endl;

    save_model(model, "test.pt");

    std::cout << "saved model" << std::endl;

    
    
}