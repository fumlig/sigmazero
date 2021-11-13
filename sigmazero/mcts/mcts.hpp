#ifndef MODEL_H
#define MODEL_H

#include "node.hpp"

#include <chess/chess.hpp>
#include <memory>

#include <unordered_map>
#include <utility>
#include <sigmazero/drl/sigmanet.hpp>

namespace mcts
{
    std::shared_ptr<Node> mcts(chess::position state, int max_iter, sigmanet& network);

}



#endif /* MODEL_H */