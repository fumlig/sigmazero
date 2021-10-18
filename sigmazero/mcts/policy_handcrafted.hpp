#ifndef POLICY_HANDCRAFTED_H
#define POLICY_HANDCRAFTED_H

#include "policy.hpp"
#include <chess/chess.hpp>
#include <random>


namespace policy
{

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

#endif /* POLICY_HANDCRAFTED_H */