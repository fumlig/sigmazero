#include <unordered_map>
#include <array>
#include <stdexcept>

#include "rules.hpp"


static const std::array<chess::direction, 8> sliding_directions =
{
    chess::direction_n,
    chess::direction_e,
    chess::direction_s,
    chess::direction_w,
	chess::direction_ne,
    chess::direction_se,
    chess::direction_sw,
    chess::direction_nw,
};

static const std::array<chess::direction, 8> knight_directions =
{
	chess::direction_nne,
    chess::direction_ene,
    chess::direction_ese,
    chess::direction_sse,
    chess::direction_ssw,
    chess::direction_wsw,
    chess::direction_wnw,
    chess::direction_nnw,
};

static const std::array<chess::direction, 3> underpromotion_directions =
{
    chess::direction_w,
    chess::direction_n,
    chess::direction_e,
};

static const std::array<chess::piece, 3> underpromotion_pieces =
{
    chess::piece_rook,
    chess::piece_knight,
    chess::piece_bishop,
};


struct move_hash
{
    std::size_t operator()(const chess::move& m) const noexcept
    {
        return ((m.from & 0xFF) << 0) | ((m.to & 0xFF) << 8) | ((m.promote & 0xFF) << 16);
    }
};

struct move_pred
{
    bool operator()(const chess::move& m1, const chess::move& m2) const noexcept
    {
        return m1.from == m2.from && m1.to == m2.to && m1.promote == m2.promote;
    }
};


static std::unordered_map<chess::move, int, move_hash, move_pred> move_to_action_mapping;
static std::unordered_map<int, chess::move> action_to_move_mapping;


static void init_action_mappings()
{
    static bool done = false;

    if(done)
    {
        return;
    }

    int action = 0;

    for(int from = chess::square_a1; from <= chess::square_h8; from++)
    {
        // sliding
        for(int i = 0; i < 8; i++)
        {
            int direction = sliding_directions[i];
            
            for(int j = 0; j < 7; j++)
            {
                int magnitude = j+1;
                int to = from + direction*magnitude;

                chess::move move(static_cast<chess::square>(from), static_cast<chess::square>(to));

                move_to_action_mapping[move] = action;
                action_to_move_mapping[action] = move;

                action++;
            }
        }

        // knight
        for(int i = 0; i < 8; i++)
        {
            int direction = knight_directions[i];
            int to = from + direction;

            chess::move move(static_cast<chess::square>(from), static_cast<chess::square>(to));

            move_to_action_mapping[move] = action;
            action_to_move_mapping[action] = move;

            action++;
        }

        // pawn
        for(int i = 0; i < 3; i++)
        {
            int direction = underpromotion_directions[i];

            for(int j = 0; j < 3; j++)
            {
                int to = from + direction;
                int promote = underpromotion_pieces[j];

                chess::move move(static_cast<chess::square>(from), static_cast<chess::square>(to), static_cast<chess::piece>(promote));

                move_to_action_mapping[move] = action;
                action_to_move_mapping[action] = move;

                action++;
            }
        }
    }

    done = true;
}


int move_action(chess::move move, chess::side turn)
{
    init_action_mappings();
    
    if(turn == chess::side_black)
    {
        move.from = chess::flip(move.from);
        move.to = chess::flip(move.to);
    }

    return move_to_action_mapping[move];
}

chess::move action_move(int action, chess::side turn)
{
    init_action_mappings();

    chess::move move = action_to_move_mapping[action];

    if(turn == chess::side_black)
    {
        move.from = chess::flip(move.from);
        move.to = chess::flip(move.to);
    }

    return move;
}


static torch::Tensor bitboard_plane(chess::bitboard bb)
{
    torch::Tensor plane = torch::zeros({8, 8});

    for(chess::square sq: chess::set_elements(bb))
    {
        int f = chess::file_of(sq);
        int r = chess::rank_of(sq);

        using namespace torch::indexing;
        plane.index_put_({r, f}, 1.0f);
    }

    return plane;
}


torch::Tensor game_image(const chess::game& game, int history)
{
    using namespace torch::indexing;

    chess::game scratch = game;
    const chess::position& position = game.get_position();
    const chess::board& board = position.get_board();

    const int planes = feature_planes*history + constant_planes;
    torch::Tensor input = torch::zeros({planes, 8, 8});

    int j = 0;
    chess::side p1 = position.get_turn();
    chess::side p2 = chess::opponent(p1);

    // feature planes
    int n = std::min(history, static_cast<int>(scratch.size()+1));
    bool flip = p1 == chess::side_black;

    for(int i = 0; i < n; i++)
    {
        // p1 pieces
        for(int p = chess::piece_pawn; p <= chess::piece_king; p++)
        {
            chess::bitboard bb = board.piece_set(static_cast<chess::piece>(p), p1);
            torch::Tensor plane = bitboard_plane(bb);
            if(flip) plane = torch::flipud(plane);

            input.index_put_({j++}, plane);
        }

        // p2 pieces
        for(int p = chess::piece_pawn; p <= chess::piece_king; p++)
        {
            chess::bitboard bb = board.piece_set(static_cast<chess::piece>(p), p2);
            torch::Tensor plane = bitboard_plane(bb);
            if(flip) plane = torch::flipud(plane);

            input.index_put_({j++}, plane);
        }

        // repetitions
        int repetitions = scratch.get_repetitions();

        if(repetitions >= 1)
        {
            input.index_put_({j}, torch::ones({8, 8}));
        } j++;

        if(repetitions >= 2)
        {
            input.index_put_({j}, torch::ones({8, 8}));
        } j++;

        // previous position (if not at initial, in which case the loop will end)
        if(!scratch.empty())
        {
            scratch.pop();
        }
    }

    j = feature_planes*history;

    // constant planes

    // color
    input.index_put_({j++}, static_cast<int>(p1));

    // move count
    input.index_put_({j++}, position.get_fullmove());

    // p1 castling
    input.index_put_({j++}, position.can_castle_kingside(p1));
    input.index_put_({j++}, position.can_castle_queenside(p1));

    // p2 castling
    input.index_put_({j++}, position.can_castle_kingside(p2));
    input.index_put_({j++}, position.can_castle_queenside(p2));

    // no-progress count
    input.index_put_({j++}, position.get_halfmove_clock());

    return input;

}

sigmanet make_network(int history)
{
    return sigmanet(feature_planes*history + constant_planes, 128, 10);
}