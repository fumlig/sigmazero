#include <chess/chess.hpp>
#include <mcts/node.hpp>
#include <mcts/policy_handcrafted.hpp>
#include <mcts/rollout.hpp>
#include <mcts/mcts_model.hpp>
#include <eval-handcrafted/eval.hpp>
#include <iostream>
#include <string>
#include <memory>
#include <unordered_map>

// Usage: ./main MCTS_ITER <config filename>

int main(int argc, char* argv[])
{
    std::string config_file_name{"mcts_config.txt"};
    if(argc > 1) config_file_name = argv[1];
    std::unordered_map<std::string, int> dict = parse_config(config_file_name);
    int MAX_MOVES = dict["MAX_MOVES"];
    int MAX_MCTS_ITERATIONS = dict["MAX_MCTS_ITERATIONS"];
    int PRINT_TIME = dict["PRINT_TIME"];
    int ROLLOUT_SIMULATIONS = dict["ROLLOUT_SIMULATIONS"];
    int WIN_SCORE = dict["WIN_SCORE"];
    int DRAW_SCORE = dict["DRAW_SCORE"];

    // Initialize engine & set node parameters
    chess::init();
    node::init(WIN_SCORE, DRAW_SCORE, 2.0);

    // Initialize random generator
    static std::random_device random_device;
    static std::mt19937 generator(random_device());
    // Set rollout policy
    // Initialize MCTS node
    chess::side enemy_side = chess::side::side_black;
    chess::position game_board = chess::position::from_fen(chess::position::fen_start); // Or chess::position::from_fen("3K4/8/8/8/8/6R1/7R/3k4 w - - 0 1")
    
    policy::HandcraftedPolicy handcrafted{generator}, handcrafted_deterministic{generator, true};
    policy::RandomPolicy random{generator};

    rollout::AveragedRollout averaged_rollout{ROLLOUT_SIMULATIONS};

    // Different models
    mcts_model::TimedModel model_handcrafted{[](const chess::position& state_, chess::side, policy_type){
        chess::position state = state_;
        double val = eval::evaluate(state);
        if (val == abs(eval::evaluator::infinity())) {
            return val < 0 ? -1.0 : 1.0;
        }
        return val / (2*eval::evaluator::MATERIAL_MAX);
    }, handcrafted, chess::side::side_white};

    mcts_model::TimedModel model_mcts_handcrafted{averaged_rollout, handcrafted, chess::side::side_white};
    mcts_model::TimedModel model_mcts_random{averaged_rollout, random, chess::side::side_black};

    // Initialize models to play against each other
    mcts_model::TimedModel model_1 = model_mcts_handcrafted;
    mcts_model::TimedModel model_2 = model_mcts_random;
    
    short moves{0};

    while(true)
    {
        chess::move model_1_move{model_1.search(game_board, MAX_MCTS_ITERATIONS)};
        game_board.make_move(model_1_move);
        if(game_board.is_checkmate() || game_board.is_stalemate()) break;
        chess::move model_2_move{model_2.search(game_board, MAX_MCTS_ITERATIONS)};
        game_board.make_move(model_2_move);
        if(game_board.is_checkmate() || game_board.is_stalemate() || moves++ == MAX_MOVES) break; 
        std::cout << "player 1 move " << model_1_move.to_lan() << std::endl;
        std::cout << "player 2 move " << model_2_move.to_lan() << std::endl;
        std::cout << "-- Game state --" << std::endl << game_board.to_string() << std::endl << std::endl;
    }
    
    std::cout << "-- Final state --" << std::endl << game_board.to_string() << std::endl;
    std::cout << "time report for model 1:" << std::endl << model_1.time_report() << std::endl;
    std::cout << "time report for model 2:" << std::endl << model_2.time_report() << std::endl;
    std::cout <<  (game_board.is_checkmate() ? "Checkmate" : "Draw") << std::endl;
    return 0;
}
