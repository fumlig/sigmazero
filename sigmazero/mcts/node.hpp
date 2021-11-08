#ifndef NODE_H
#define NODE_H

#include "network.hpp"
#include <chess/chess.hpp>
#include "misc.hpp"
#include <vector>
#include <float.h>
#include <memory>

namespace mcts
{    
class Node : public std::enable_shared_from_this<Node>
{

    public:
        // Used to create a node that is not a parent node
        Node(chess::position state, bool is_start_node, std::weak_ptr<Node> parent, chess::move move);
        // Used to create a parent node
        Node(chess::position state);
        ~Node() = default;
        // Get child nodes
        std::vector<std::shared_ptr<Node>> get_children() const;
        // Backpropagate score and visits to parent node
        void backpropagate(double value);
        // Expand node
        void expand(const std::unordered_map<size_t, double>& action_probabilities);
        void explore_and_set_priors(const Network &network);
        void add_exploration_noise(double dirichlet_alpha, double exploration_factor);
        double get_value() const;
        double get_terminal_value() const;


        // UCB1 scoring function
        double UCB1() const;
        // Determine next node to expand/rollout by traversing tree
        std::shared_ptr<Node> traverse();
        // Retrieve the best child node based on UCB1 score
        // Can be useful if we want to keep the tree from the previous iterations
        std::shared_ptr<Node> best_child() const;
        // Get action distribution for the children of this node.
        // Should be ran after the entire mcts search is completeted.
        std::vector<double> action_distribution(size_t num_actions);
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
        static double pb_c_base;
        static double pb_c_init;

    private:

        bool is_terminal_node = false;
        double t = 0.0;
        int n = 0;

        chess::position state;
        bool is_start_node;
        std::weak_ptr<Node> parent;
        chess::move move;

        std::vector<std::shared_ptr<Node>> children;
        double prior;
        size_t action;
};

} //namespace node

#endif /* NODE_H */