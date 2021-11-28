#include <algorithm>
#include <cstdint>
#include <cmath>
#include <chess/chess.hpp>
#include <iostream>

#include "sigmanet.hpp"
#include "rules.hpp"


residual_block::residual_block(int filters) {

    conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(filters, filters, 3).padding(1));
    batchnorm1 = torch::nn::BatchNorm2d(filters);

    conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(filters, filters, 3).padding(1));
    batchnorm2 = torch::nn::BatchNorm2d(filters);

    register_module("conv1", conv1);
    register_module("batchnorm1", batchnorm1);
    register_module("conv2", conv2);
    register_module("batchnorm2", batchnorm2);
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

sigmanet_impl::sigmanet_impl(int channels, int filters, int blocks) : channels{channels}, filters{filters}, blocks{blocks} {

    input_conv = torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, filters, 3).stride(1).padding(1)),
        torch::nn::BatchNorm2d(filters),
        torch::nn::ReLU()
    );

    residual = torch::nn::Sequential();

    for (int i = 0; i < blocks; i++) {
        residual->push_back(residual_block(filters));
    }

    value_head = torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(filters, 1, 1)),
        torch::nn::BatchNorm2d(1),
        torch::nn::ReLU(),
        torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(-2).end_dim(-1)),
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
        torch::nn::Linear(2 * 8 * 8, 8 * 8 * 73)
    );

    register_module("input_conv", input_conv);
    register_module("residual", residual);
    register_module("value_head", value_head);
    register_module("policy_head", policy_head);
}


std::pair<torch::Tensor, torch::Tensor> sigmanet_impl::forward(torch::Tensor x) {
    x = input_conv->forward(x);
    x = residual->forward(x);

    auto value = value_head->forward(x);
    auto policy = policy_head->forward(x);//torch::softmax(policy_head->forward(x), -1);

    return std::make_pair(value, policy);
}



// z is model output value, v is mcts value, p is model output policy, pi is mcts policy
torch::Tensor sigma_loss(torch::Tensor z, torch::Tensor v, torch::Tensor p, torch::Tensor pi) {
    //p = torch::add(p, 1e-8);
    torch::Tensor value_loss = torch::mse_loss(z.squeeze(), v, torch::Reduction::Sum);
    torch::Tensor policy_loss = torch::cross_entropy_loss(p, pi, {}, torch::Reduction::Sum);
    //torch::Tensor policy_loss = -torch::sum(torch::mul(pi, torch::log(p)));
    torch::Tensor loss = value_loss + policy_loss;

    return loss;
}
