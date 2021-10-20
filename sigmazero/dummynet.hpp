#ifndef SIGMAZERO_DUMMYNET_HPP
#define SIGMAZERO_DUMMYNET_HPP


#include <torch/torch.h>

// https://pytorch.org/tutorials/advanced/cpp_frontend.html
// - TORCH_MODULE[_IMPL] is required for torch::{save,load} to work
// - register_{module,parameter} are also required in the implementation

struct dummynet_impl: torch::nn::Module {};
TORCH_MODULE_IMPL(dummynet, dummynet_impl);


#endif