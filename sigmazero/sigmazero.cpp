#include <string>
#include <filesystem>
#include <random>
#include <chrono>
#include <sstream>

#include <chess/chess.hpp>
#include <uci/uci.hpp>
#include <torch/torch.h>

#include "sigmanet.hpp"
#include "search.hpp"
#include "rules.hpp"


class sigmazero: public uci::engine
{
private:
    sigmanet model;
    torch::Device device;
    chess::game game;
    
public:
    sigmazero(sigmanet model, torch::Device device):
    uci::engine(),
    model(model),
    device(device),
    game()
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
        std::cerr << game.to_string() << std::endl;
    }

    uci::search_result search(const uci::search_limit& limit, uci::search_info& info, const std::atomic_bool& ponder, const std::atomic_bool& stop) override
    {
        long simulations = 0;
        auto start_time = std::chrono::steady_clock::now();

        chess::side turn = game.get_position().get_turn();
        float clock = limit.clocks[turn];
        float increment = limit.increments[turn];

        // https://chess.stackexchange.com/questions/2506/what-is-the-average-length-of-a-game-of-chess
        int ply = game.size();
        int remaining_halfmoves = 59.3 + (72830 - 2330*ply)/(2644 + ply*(10 + ply));
        float budgeted_time = clock/remaining_halfmoves; // todo: increment

        info.message("budgeted time: " + std::to_string(budgeted_time));
        info.message("starting simulations");

        auto stop_search = [&](const node& root)
        {
            if(stop)
            {
                return true;
            }

            auto current_time = std::chrono::steady_clock::now();
            float elapsed_time = std::chrono::duration<float>(current_time - start_time).count();
            
            bool unlimited = limit.infinite || ponder; 

            if(!unlimited)
            {
                if(elapsed_time > limit.time)
                {
                    info.message("stopping search due to time limit");
                    return true;
                }

                if(elapsed_time > budgeted_time)
                {
                    info.message("stopping search due to budgeted time exceeded");
                    return true;
                }
            }

            std::shared_ptr<node> best = root.select_best();

            if(best)
            {
                std::ostringstream child_visits;
                for(std::shared_ptr<node> child: root.children) child_visits << child->move.to_lan() << ' ' << child->visit_count << ' ';

                info.nodes(simulations);
                info.score(best->value());
                info.line({best->move});
                info.message("value " + std::to_string(best->value()));
                info.message("visits " + child_visits.str());
            }

            simulations++;

            return false;
        };

        std::shared_ptr<node> best = run_mcts(game, model, device, stop_search, false);
        std::shared_ptr<node> next = best->select_best();

        uci::search_result result;
        result.best = best->move;
        if(next) result.ponder = next->move;

        return result;
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
    sigmanet model = make_network();

    torch::load(model, model_path);

    model->to(device);
    model->eval();
    model->zero_grad();

    sigmazero engine(model, device);
    
    return uci::main(engine);
}
