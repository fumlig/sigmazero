#ifndef SIGMANET_H
#define SIGMANET_H

#include <torch/torch.h>
#include <utility>

class residual_block : public torch::nn::Module {

    torch::nn::Conv2d conv1;
    torch::n::BatchNorm2d batchnorm1;
    torch::nn::Conv2d conv2;
    torch::n::BatchNorm2d batchnorm2;

public:

    residual_block(int filters);

    torch::Tensor forward(torch::Tensor x);

}


class sigmanet : public torch::nn::Module {

    torch::nn::Sequential input_conv;
    torch::nn::Sequential residual;
    torch::nn::Sequential value_head;
    torch::nn::Sequential policy_head;

public:

    sigmanet(int filters, int blocks);

    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x);
};

#endif