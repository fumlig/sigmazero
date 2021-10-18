#include "sigmanet.hpp"

residual_block::residual_block(int filters) {
    
    conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(filters, filters, 3));
    batchnorm1 = torch::nn::BatchNorm2d(filters);

    conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(filters, filters, 3));
    batchnorm2 = torch::nn::BatchNorm2d(filters);

    register_module("conv1", conv1);
    register_module("batchnorm1", batchnorm1);
    register_module("conv2", conv2);
    register_module("conv2", batchnorm2);
}

torch::Tensor residual_block::forward(torch::Tensor x) {
    
    torch::Tensor y = x;

    x = conv1->forward(x);
    x = batchnorm1->forward(x);
    
    x = torch::relu(x);

    x = conv2->forward(x);
    x = batchnorm2->forward(x);

    x = y + x;
    x = torch::relu(x);

    return x;
}

sigmanet::sigmanet(int channels, int filters, int blocks) {

    input_conv = torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, filters, 3).stride(1).padding(1)),
        torch::nn::BatchNorm2d(filters),
        torch::nn::ReLU()
    );

    for (int i = 0; i < blocks; i++) {
        residual->push_back(residual_block(filters));
    }

    value_head = torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(filters, 1, 1)),
        torch::nn::BatchNorm2d(1),
        torch::nn::ReLU(),
        torch::nn::Linear(8 * 8, 256),
        torch::nn::ReLU(),
        torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(-2).end_dim(-1)),
        torch::nn::Linear(256, 1),
        torch::nn::Tanh()
    );

    policy_head = torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(filters, 2, 1)),
        torch::nn::BatchNorm2d(2),
        torch::nn::ReLU(),
        torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(-3).end_dim(-1)),
        torch::nn::Linear(2 * 8 * 8, 1337) // Change 1337 to number of output moves
    );

    register_module("input_conv", input_conv);
    register_module("residual", residual);
    register_module("value_head", value_head);
    register_module("policy_head", policy_head);
}

std::pair<torch::Tensor, torch::Tensor> sigmanet::forward(torch::Tensor x) {

    x = input_conv->forward(x);
    x = residual->forward(x);

    auto value = value_head->forward(x);
    auto policy = policy_head->forward(x);

    return std::make_pair(value, policy);
}

torch::Tensor sigma_loss(torch::Tensor z, torch::Tensor v, torch::Tensor pi, torch::Tensor p) {
    torch::Tensor value_loss = torch::mean((z-v)*(z-v));
    torch::Tensor policy_loss = torch::dot(pi, torch::log(p));
    torch::Tensor loss = torch::add(value_loss, policy_loss);
    return loss;
}