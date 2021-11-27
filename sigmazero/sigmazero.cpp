#include <string>
#include <filesystem>
#include <random>
#include <chrono>

#include <chess/chess.hpp>
#include <uci/uci.hpp>
#include <torch/torch.h>

#include "drl/sigmanet.hpp"
#include "mcts/node.hpp"


class sigmazero: public uci::engine
{
private:
    sigmanet model;
    torch::Device device;
    
    chess::game game;
    std::shared_ptr<mcts::Node> node;

    const float& pb_c_base;
    const float& pb_c_init;

public:
    sigmazero(sigmanet model, torch::Device device):
    uci::engine(),
    model(model),
    device(device),
    game(),
    node(),
    pb_c_base{opt.add<uci::option_float>("PB C Base", 19652.0f).ref()},
    pb_c_init{opt.add<uci::option_float>("PB C Init", 1.25f).ref()}
    {

    }

    ~sigmazero()
    {

    }

    std::string name() const override
    {
        return "sigmazero";
    }

    std::string author() const override
    {
        return "Erik, Oskar, Justus, Bence";
    }

    void setup(const chess::position& position, const std::vector<chess::move>& moves) override
    {
        game = chess::game(position, moves);
        node = std::make_shared<mcts::Node>(game.get_position());
    }

    uci::search_result search(const uci::search_limit& limit, uci::search_info& info, const std::atomic_bool& ponder, const std::atomic_bool& stop) override
    {
        auto start_time = std::chrono::steady_clock::now();

        const chess::position& position = game.get_position();
        chess::side turn = position.get_turn();

        float clock = limit.clocks[turn];
        float increment = limit.increments[turn];

        // https://chess.stackexchange.com/questions/2506/what-is-the-average-length-of-a-game-of-chess
        int ply = game.size();
        int remaining_halfmoves = 59.3 + (72830 - 2330*ply)/(2644 + ply*(10 + ply));
        float budgeted_time = (clock + increment*remaining_halfmoves)/remaining_halfmoves;

        info.message("budgeted time: " + std::to_string(budgeted_time));

        mcts::Node::pb_c_base = pb_c_base;
        mcts::Node::pb_c_init = pb_c_init;

        auto evaluation = model->evaluate(game.get_position(), device);
        
        node->explore_and_set_priors(evaluation);

        unsigned simulations = 0;

        info.message("starting simulations");

        while(!stop)
        {
            auto current_time = std::chrono::steady_clock::now();
            float elapsed_time = std::chrono::duration<float>(current_time - start_time).count();
            
            bool unlimited = limit.infinite || ponder; 

            if(!unlimited)
            {
                if(elapsed_time > limit.time)
                {
                    info.message("stopping search due to time limit");
                    break;
                }

                if(elapsed_time > budgeted_time)
                {
                    info.message("stopping search due to budgeted time exceeded");
                    break;
                }
            }
        
            std::shared_ptr<mcts::Node> n = node->traverse();

            if(n->is_over())
            {
                n->backpropagate(n->get_terminal_value());
                continue;
            }
            
            evaluation = model->evaluate(n->get_state(), device);
            n->explore_and_set_priors(evaluation);

            simulations++;

            // not really correct but whatever...
            info.depth(simulations);
        }

        chess::move best, ponder_move;
        
        if(node->best_child())
        {
            best = node->best_child()->get_move();
        
            if(node->best_child()->best_child())
            {
                ponder_move = node->best_child()->best_child()->get_move();
            }
        }

        return {best, ponder_move};
    }

    void reset() override
    {

    }
};


int main(int argc, char** argv)
{
    chess::init();

    std::filesystem::path model_path = argc >= 2 ? argv[1] : "model.pt";
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    sigmanet model(0, 128, 10);

    torch::load(model, model_path);

    model->to(device);
    model->eval();
    model->zero_grad();

    sigmazero engine(model, device);
    
    return uci::main(engine);
}
