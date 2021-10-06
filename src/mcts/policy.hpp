#ifndef POLICY_H
#define POLICY_H

#include <chess/chess.hpp>
#include <functional>
#include <random>
// Stores some different policies that can be used by Node
using policy_type = std::function<chess::move(chess::position)>;

namespace policy
{

class Policy {

    public:
    virtual chess::move operator()(chess::position state) = 0;
};

class RandomPolicy : public Policy {
    std::mt19937 &generator;

    public:
    RandomPolicy(std::mt19937 &generator) : generator{generator} {}
    chess::move operator() (chess::position state) override;
};

class HandcraftedPolicy : public Policy {
    std::mt19937 &generator;
    bool deterministic;
    bool look_ahead;
    const double VALUE_SCALE = 0.01;

    public:
    HandcraftedPolicy(std::mt19937 &generator, bool deterministic=false, bool look_ahead=false) :
        generator{generator}, deterministic{deterministic}, look_ahead{look_ahead} {}
    chess::move operator() (chess::position state) override;

    private:
    std::vector<double> softmax_distribution(std::vector<double> values);
    double get_value_after_move(chess::position &state, const chess::move& move);
    double get_value(chess::position &state, const chess::move& move);
};
}

#endif /* POLICY_H */