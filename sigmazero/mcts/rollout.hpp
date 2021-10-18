#ifndef ROLLOUT_H
#define ROLLOUT_H

#include <functional>
#include <chess/chess.hpp>
#include "policy.hpp"


// Stores some different policies that can be used by Node
using rollout_type = std::function<double(const chess::position, chess::side, policy_type)>;

namespace rollout
{


class Rollout {
    public:
    virtual double operator()(const chess::position &state, chess::side player_side, policy_type policy) = 0;
};

class BadRollout : public Rollout {
    public:
    double operator()(const chess::position &state, chess::side player_side, policy_type policy) override;
};

class AveragedRollout : public Rollout {
    int n_iter;
    // Perhaps make below the global specification
    // of scores, instead of in Node::
    // Right now we get a circular import because of using header only
    // files if we use the Node::WIN_SCORE.
    double WIN_SCORE = 1.0;
    double DRAW_SCORE = 0.0;

    public:
    AveragedRollout(int n_iter=10) : n_iter{n_iter} {}
    double operator()(const chess::position &state, chess::side player_side, policy_type policy) override;
};
}

#endif // ROLLOUT_H