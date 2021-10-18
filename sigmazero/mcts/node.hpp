#ifndef NODE_H
#define NODE_H

#include <chess/chess.hpp>
#include <mcts/rollout.hpp>
#include <mcts/policy.hpp>
#include <mcts/misc.hpp>
#include <vector>
#include <float.h>
#include <memory>

namespace node
{    
class Node : public std::enable_shared_from_this<Node>
{
    public:
        // Used to create a node that is not a parent node
        Node(chess::position state, chess::side player_side, bool is_start_node, std::weak_ptr<Node> parent, chess::move move);

        // Used to create a parent node
        Node(chess::position state, chess::side player_side);
        ~Node() = default;
        // Get child nodes
        inline std::vector<std::shared_ptr<Node>> get_children() const
        {
            return this->children;
        }

        // Perform rollout from state
        void rollout(rollout_type rollout_method, policy_type policy);

        // Backpropagate score and visits to parent node
        void backpropagate();

        // Expand node
        void expand();

        // UCB1 scoring function
        inline double UCB1() const
        {
            auto p = parent.lock();
            if (n == 0 || (p && p->n==0))
            {
                return DBL_MAX;
            }
            else
            {
                double v_bar = t / n;
                int N = p ? p->n : 1;
                return v_bar + UCB1_CONST*sqrt(log(N) / n);
            }
        }

        // Determine next node to expand/rollout by traversing tree
        std::shared_ptr<Node> traverse();

        // Retrieve the best child node based on UCB1 score
        // Can be useful if we want to keep the tree from the previous iterations
        std::shared_ptr<Node> best_child() const;
        // Get the move that gives the best child
        // Useful for baseline mcts algorithm
        chess::move best_move() const;

        // Get state
        chess::position get_state() const;
        // Check if current state is a terminal state
        bool is_over() const;
        // Get amount of vists
        int get_n() const;

        // Print the main node and its children
        std::string to_string(int layers_left=1) const;

        static double WIN_SCORE;
        static double DRAW_SCORE;
        static double UCB1_CONST;

    protected:
        chess::position state;
        chess::side player_side;
        chess::move move;
        bool is_start_node;
        bool is_terminal_node = false;
        std::weak_ptr<Node> parent;
        std::vector<std::shared_ptr<Node>> children;
        double t;
        int n;
};

// Initialize node library 
// Sets reward scores
// Not necessary unless modifying scores is desired
void init(double win_score, double draw_score, double UCB1_const);
} //namespace node

#endif /* NODE_H */