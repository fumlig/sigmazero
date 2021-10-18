#include "./rollout.hpp"
#include <chess/chess.hpp>
#include "policy.hpp"

namespace rollout
{

double BadRollout::operator()(const chess::position &state, chess::side player_side, policy_type policy) {
    return 0.0;
}
    



double AveragedRollout::operator()(const chess::position &state, chess::side player_side, policy_type policy) {
    double accumulated_t{0};
    for (int i = 0; i < n_iter; ++i)
    {
        chess::position rollout_state = state;
        short uneventful_timer = 0;
        while (!rollout_state.is_checkmate() && !rollout_state.is_stalemate() && uneventful_timer < 50)
        {
            chess::move choice = policy(rollout_state);
            chess::undo undo = rollout_state.make_move(choice);
            uneventful_timer = undo.capture != chess::piece::piece_none ? 0 : uneventful_timer + 1;
        }

        bool is_player_turn = rollout_state.get_turn() == player_side;
        if (rollout_state.is_checkmate())
        {
            accumulated_t += is_player_turn ? -WIN_SCORE : WIN_SCORE;
        }
        else
        {
            accumulated_t += DRAW_SCORE;
        }
    }

    return accumulated_t / n_iter;
}


} // namespace rollout
