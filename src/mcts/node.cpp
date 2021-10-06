
#include "./node.hpp"
#include <chess/chess.hpp>
#include <mcts/rollout.hpp>
#include <mcts/policy.hpp>
#include <mcts/misc.hpp>
#include <vector>
#include <iterator>
#include <float.h>
#include <math.h>
#include <memory>

namespace node
{    

// Used to create a node that is not a parent node
Node::Node(chess::position state, chess::side player_side, bool is_start_node, std::weak_ptr<Node> parent, chess::move move)
    : state{state},
    player_side{player_side},
    is_start_node{is_start_node},
    parent{parent},
    move{move}
{
    this->children = {};
    this->t = 0;
    this->n = 0;
}
// Used to create a parent node
Node::Node(chess::position state, chess::side player_side) : Node(state, player_side, true, std::weak_ptr<Node>(), chess::move()) {}

// Perform rollout from state
void Node::rollout(rollout_type rollout_method, policy_type policy)
{
    t = rollout_method(state, player_side, policy);
    n++;
}

// Backpropagate score and visits to parent node
void Node::backpropagate()
{
    if (auto p = parent.lock())
    {
        p->t += t;
        p->n++;

        if (!p->is_start_node)
        {
            p->backpropagate();
        }
    }
}

// Expand node
void Node::expand()
{   
    std::vector<chess::move> available_moves{state.moves()};
    for (chess::move child_move : available_moves)
    {
        chess::position child_state = state.copy_move(child_move); // TODO - Make this optional
        std::shared_ptr<Node> new_child = std::make_shared<Node>(child_state, player_side, false, weak_from_this(), child_move);
        if (new_child->state.is_checkmate() || new_child->state.is_stalemate())
        {
            if (new_child->state.is_checkmate())
            {
                new_child->t = new_child->state.get_turn() == player_side ? -WIN_SCORE : WIN_SCORE;
            }
            else
            {
                new_child->t = DRAW_SCORE;
            }
            new_child->is_terminal_node = true;
            new_child->n = 1;
            new_child->backpropagate();
        }
        children.push_back(new_child);
    }
}


// Determine next node to expand/rollout by traversing tree
std::shared_ptr<Node> Node::traverse()
{
    std::vector<double> UCB1_scores{};
    for (std::shared_ptr<Node> child : children)
    {
        if (!child->is_terminal_node)
            UCB1_scores.push_back(child->UCB1());
    }
    if (UCB1_scores.size() == 0)
    {
        is_terminal_node = true;
        return parent.lock() ? parent.lock() : shared_from_this();
    }

    std::shared_ptr<Node> best_child = get_max_element<std::shared_ptr<Node>>(children.begin(), UCB1_scores.begin(), UCB1_scores.end());

    if (best_child->children.size() > 0)
    {
        return best_child->children.front()->traverse();
    }
    else
    {
        return best_child;
    }
}

// Retrieve the best child node based on UCB1 score
// Can be useful if we want to keep the tree from the previous iterations
std::shared_ptr<Node> Node::best_child() const
{
    std::vector<double> winrates{};
    for (std::shared_ptr<Node> child : children)
    {
        winrates.push_back(child->t);
    }
    return get_max_element<std::shared_ptr<Node>>(children.begin(), winrates.begin(), winrates.end());
}
// Get the move that gives the best child
// Useful for baseline mcts algorithm
chess::move Node::best_move() const
{
    return best_child()->move;
}

// Get state
chess::position Node::get_state() const
{
    return state;
}

// Check if current state is a terminal state
bool Node::is_over() const
{
    return is_terminal_node || state.is_checkmate() || state.is_stalemate();
}

// Get amount of vists
int Node::get_n() const
{
    return n;
}

// Print the main node and its children
std::string Node::to_string(int layers_left) const
{
    std::string tree{};
    tree += state.pieces().to_string();

    if(layers_left > 0) {
        tree += '\n' + "---children depth " + std::to_string(layers_left) + " ---\n";
        for(std::shared_ptr<Node> child_ptr : children)
        {
            tree += child_ptr->to_string(layers_left-1) + '\n';
        }

    }
    return tree;
}


double Node::WIN_SCORE = 1.0;
double Node::DRAW_SCORE = 0.0;
double Node::UCB1_CONST = 2.0;

void init(double win_score, double draw_score, double UCB1_const) {
    Node::WIN_SCORE = win_score;
    Node::DRAW_SCORE = draw_score;
    Node::UCB1_CONST = UCB1_const;
};

}

