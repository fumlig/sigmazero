#ifndef SIGMAZERO_DUMMYNET_HPP
#define SIGMAZERO_DUMMYNET_HPP


#include <torch/torch.h>
#include <cstdint>

// https://pytorch.org/tutorials/advanced/cpp_frontend.html
// - TORCH_MODULE[_IMPL] is required for torch::{save,load} to work
// - register_{module,parameter} are also required in the implementation

struct dummynet_impl: torch::nn::Module
{
	dummynet_impl(std::int64_t N, std::int64_t M):
	linear(register_module("linear", torch::nn::Linear(N, M)))
	{
		bias = register_parameter("bias", torch::randn(M));
	}

	torch::Tensor forward(torch::Tensor input)
	{
		return linear(input) + bias;
	}
	torch::nn::Linear linear;
	torch::Tensor bias;
};

TORCH_MODULE_IMPL(dummynet, dummynet_impl);


#endif