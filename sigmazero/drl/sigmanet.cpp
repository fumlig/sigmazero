#include <algorithm>
#include <cstdint>

#include "sigmanet.hpp"


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

sigmanet::sigmanet(int history, int filters, int blocks) : history{history}, in_channels{history*feature_planes + constant_planes}, filters{filters}, blocks{blocks} {

    input_conv = torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, filters, 3).stride(1).padding(1)),
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


std::pair<torch::Tensor, torch::Tensor> sigmanet::forward(torch::Tensor x) {

    x = input_conv->forward(x);
    x = residual->forward(x);

    auto value = value_head->forward(x);
    auto policy = torch::softmax(policy_head->forward(x), -1);

    return std::make_pair(value, policy);
}


torch::Tensor sigmanet::encode_input(const chess::game& g)
{
    using namespace torch::indexing;

    const int planes = in_channels;
    torch::Tensor input = torch::zeros({planes, 8, 8});

    int j = 0;
    const chess::position& p = g.get_position();

    chess::game h = g;

    chess::side p1 = p.get_turn();
    chess::side p2 = chess::opponent(p1);

    // feature planes
    bool flip = p1 == chess::side_black;
    int n = std::min(history, static_cast<int>(h.size()+1));
    
    for(int i = 0; i < n; i++)
    {
        // p1 pieces
        for(int p = chess::piece_pawn; p <= chess::piece_king; p++)
        {
            chess::bitboard bb = h.get_position().get_board().piece_set(static_cast<chess::piece>(p), p1);
            torch::Tensor plane = bitboard_plane(bb);
            if(flip) plane = torch::flipud(plane);

            input.index_put_({j++}, plane);
        }

        // p2 pieces
        for(int p = chess::piece_pawn; p <= chess::piece_king; p++)
        {
            chess::bitboard bb = h.get_position().get_board().piece_set(static_cast<chess::piece>(p), p2);
            torch::Tensor plane = bitboard_plane(bb);
            if(flip) plane = torch::flipud(plane);

            input.index_put_({j++}, plane);
        }

        // repetitions
        int repetitions = h.get_repetitions();

        if(repetitions >= 1)
        {
            input.index_put_({j}, torch::ones({8, 8}));
        } j++;

        if(repetitions >= 2)
        {
            input.index_put_({j}, torch::ones({8, 8}));
        } j++;

        // previous position (if not at initial, in which case the loop will end)
        if(!h.empty())
        {
            h.pop();
        }
    }

    j = feature_planes*history;

    // constant planes

    // color
    input.index_put_({j++}, static_cast<int>(p1));

    // move count
    input.index_put_({j++}, p.get_fullmove());

    // p1 castling
    input.index_put_({j++}, p.can_castle_kingside(p1));
    input.index_put_({j++}, p.can_castle_queenside(p1));

    // p2 castling
    input.index_put_({j++}, p.can_castle_kingside(p2));
    input.index_put_({j++}, p.can_castle_queenside(p2));

    // no-progress count
    input.index_put_({j++}, p.get_halfmove_clock());

    return input;
}

int sigmanet::get_input_channels() const
{
    return in_channels;
}

torch::Tensor sigma_loss(torch::Tensor z, torch::Tensor v, torch::Tensor pi, torch::Tensor p) {

    torch::Tensor value_loss = torch::sum(torch::mul(z-v, z-v));
    torch::Tensor policy_loss = torch::sum(torch::mul(pi, torch::log(p)));
    torch::Tensor loss = value_loss + policy_loss;
    return loss;
}


torch::Tensor bitboard_plane(chess::bitboard bb)
{
    torch::Tensor plane = torch::zeros({8, 8});

    for(chess::square sq: chess::set_elements(bb))
    {
        chess::file f = chess::file_of(sq);
        chess::rank r = chess::rank_of(sq);

        using namespace torch::indexing;
        plane.index_put_({static_cast<int>(r), static_cast<int>(f)}, 1.0f);
    }

    return plane;
}


