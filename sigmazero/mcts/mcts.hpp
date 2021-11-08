#ifndef MODEL_H
#define MODEL_H

#include "node.hpp"
#include "network.hpp"

#include <chess/chess.hpp>
#include <memory>

namespace mcts
{
    std::shared_ptr<Node> mcts(chess::position state, int max_iter, const mcts::Network& network);

}



#endif /* MODEL_H */