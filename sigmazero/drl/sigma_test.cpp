#include <iostream>

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
    torch::Device device(torch::kCPU);
    // Check cuda support
    if(torch::cuda::is_available())
    {
        device = torch::Device(torch::kCUDA);
        std::cout << "CUDA is available - using it." << std::endl;
    }

    int n_filters = 64;
    int n_blocks = 10;
    double c = 0.0001; // L2 Regularization
    // Create dummy input data
    long batch_size = 64;
    int history = 2;
    int n_moves = 8 * 8 * 73; // Change in sigmanet as well


    torch::Tensor input_state = torch::rand({batch_size, in_channels, 8, 8}, device);
    // Create dummy output data
    torch::Tensor value_targets = torch::rand({batch_size, 1}, device);
    torch::Tensor policy_targets = torch::rand({batch_size, n_moves}, device);

    std::cout << "randomised data" << std::endl;

    // Initialize model loss and optimizer
    sigmanet model(history, n_filters, n_blocks);

    std::cout << "initialised model" << std::endl;

    model.to(device);
    model.train();
    // TODO: Move model to gpu if available?

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

    // Inference
    model.eval();
    torch::Tensor test_state = torch::rand({1, in_channels, 8, 8}, device);
    auto[value, policy] = model.forward(test_state);

    std::cout << "inference result: " << std::endl << "value: " << value << std::endl << "policy: " << policy << std::endl;

    save_model(model, "test.pt");

    std::cout << "saved model" << std::endl;

}