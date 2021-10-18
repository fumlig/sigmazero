#include "policy_handcrafted.hpp"

#include <chess/chess.hpp>
#include "misc.hpp"
#include <eval/eval.hpp>

#include <random>
#include <algorithm>


namespace policy
{

chess::move HandcraftedPolicy::operator() (chess::position state) {
    std::vector<chess::move> moves{state.moves()};
    std::vector<double> move_values(moves.size());
    double exp_sum = 0.0;
    for (size_t i = 0; i < moves.size(); i++) {
        // double val = 0.0; // Random

        move_values[i] = look_ahead ? 
                        get_value_after_move(state, moves[i]) : 
                        get_value(state, moves[i]);
        if (move_values[i] == eval::evaluator::infinity()) {
            return moves[i];
        }
    }
    if (deterministic) {
        return moves[get_max_idx(move_values.begin(), move_values.end())];
    }
    // Apply softmax
    std::vector<double> prob_distribution = softmax_distribution(move_values);

    std::uniform_real_distribution<double> unif(0, 1);
    double random = unif(generator);
    size_t idx = std::lower_bound(prob_distribution.begin(), prob_distribution.end(), random) - prob_distribution.begin();
    return moves[idx];
}

std::vector<double> HandcraftedPolicy::softmax_distribution(std::vector<double> values) {
    auto exp_scale = [&](double value) { return exp(VALUE_SCALE*value); };
    std::transform(values.begin(), values.end(), values.begin(), exp_scale);

    double exp_sum = std::accumulate(values.begin(), values.end(), 0.0);

    // Cumulative distribution
    std::partial_sum(values.begin(), values.end(), values.begin());

    auto divide_by_sum =[exp_sum](double value){ return value / exp_sum; };
    std::transform(values.begin(), values.end(), values.begin(), divide_by_sum);
    return values;
}

double HandcraftedPolicy::get_value_after_move(chess::position &state, const chess::move& move){
    chess::undo undo = state.make_move(move);
    std::vector<chess::move> opponent_moves{state.moves()};
    if (opponent_moves.empty()) {
        // Stalemate or checkmate
        double val = -eval::evaluate(state);
        state.undo_move(move, undo);
        return val; 
    }
    double min_val = 999999999; 
    for (const chess::move &opponent_move: opponent_moves) {
        chess::undo opponent_undo = state.make_move(opponent_move);
        double val = eval::evaluate(state);
        // Opponent wants to minimize the state value after it has made its move
        min_val = std::min(min_val, val);
        state.undo_move(opponent_move, opponent_undo);
    }
    state.undo_move(move, undo);
    return min_val;
}

double HandcraftedPolicy::get_value(chess::position &state, const chess::move& move){
    chess::undo undo = state.make_move(move);
    double val = -eval::evaluate(state);
    state.undo_move(move, undo);
    return val; 
}


} // namespace policy