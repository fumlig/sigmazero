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

    std::random_device random;
    std::default_random_engine generator;
    
    chess::game game;
    std::shared_ptr<mcts::Node> node;

public:
    sigmazero(const std::filesystem::path& model_path):
    model(0, 64, 13),
    device(torch::kCPU),
    random(),
    generator(random()),
    game(),
    node()
    {
        torch::load(model, model_path);

        if(torch::cuda::is_available())
        {
            device = torch::Device(torch::kCUDA);
        }

        model->to(device);
        model->eval();
        model->zero_grad();

        opt.add<uci::option_spin>("MultiPV", 1, 1, 1);
        opt.add<uci::option_spin>("Move Overhead", 0, 0, 1);
        opt.add<uci::option_spin>("Threads", 1, 1, 1);
        opt.add<uci::option_spin>("Hash", 1, 1, 1);

        opt.add<uci::option_range<unsigned>>("Sampling Moves", 30);
        opt.add<uci::option_range<float>>("Dirichlet Alpha", 0.3f, 0.0f, 1.0f);
        opt.add<uci::option_range<float>>("Exploration Fraction", 0.25f, 0.0f, 1.0f);
        opt.add<uci::option_range<float>>("PB C Base", 19652.0f);
        opt.add<uci::option_range<float>>("PB C Init", 1.25f);
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

        float dirichlet_alpha = opt.get<uci::option_range<float>>("Dirichlet Alpha");
        float exploration_fraction = opt.get<uci::option_range<float>>("Exploration Fraction");

        mcts::Node::pb_c_base = opt.get<uci::option_range<float>>("PB C Base");
        mcts::Node::pb_c_init = opt.get<uci::option_range<float>>("PB C Init");

        auto evaluation = model->evaluate(game.get_position(), device);
        
        info.message("passed evaluate");

        node->explore_and_set_priors(evaluation);
        node->add_exploration_noise(dirichlet_alpha, exploration_fraction, generator);

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

        chess::move best = node->best_child()->get_move();
        chess::move ponder_move = node->best_child()->best_child()->get_move();

        return {best, ponder_move};
    }

    void reset() override
    {

    }
};


int main(int argc, char** argv)
{
    std::filesystem::path model_path = "model.pt";

    if(argc >= 2)
    {
        model_path = argv[1];
    }

    chess::init();
    
    sigmazero engine(model_path);
    
    return uci::main(engine);
}
