#ifndef MODEL_H
#define MODEL_H

#include <mcts/node.hpp>
#include <mcts/misc.hpp>
#include <mcts/rollout.hpp>
#include <mcts/policy.hpp>
#include <chess/chess.hpp>
#include <memory>
#include <string>

namespace mcts_model
{
    
    // Model struct to simplify usage of the search
    struct Model
    {
        Model(rollout_type rollout, policy_type policy, chess::side model_side);

        virtual chess::move search(chess::position state, int max_iter);
        rollout_type rollout;
        policy_type policy;
        chess::side model_side;
    };
    // Tracks time spent on different steps of MCTS search
    struct TimedModel : public Model
    {
        TimedModel(rollout_type rollout, policy_type policy, chess::side model_side);

        chess::move search(chess::position state, int max_iter) override;


        Timer inner_timer{};
        Timer outer_timer{};
        double t_expanding{0};
        double t_traversing{0};
        double t_rollouting{0};
        double t_backpropping{0};
        double t_tot{0};

        std::string time_report();
    };
}



#endif /* MODEL_H */