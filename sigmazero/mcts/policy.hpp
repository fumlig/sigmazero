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
}

#endif /* POLICY_H */