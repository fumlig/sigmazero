#include <unordered_map>
#include <array>
#include <stdexcept>

#include "rules.hpp"


int move_action(chess::move move, const chess::game& game)
{
    const chess::position& position = game.get_position();
    const chess::board& board = position.get_board();

    const int underpromotion_type = 0;
    const int knight_type = underpromotion_type + underpromotion_actions;
    const int sliding_type = knight_type + knight_actions;

    auto [side, piece] = board.get(move.from);

    if(position.get_turn() == chess::side_black)
    {
        move.from = chess::flip(move.from);
        move.to = chess::flip(move.to);
    }

    int square = move.from;
    int type = -1;

    if(piece == chess::piece_pawn && chess::rank_of(move.to) == chess::rank_8 && move.promote != chess::piece_queen)
    {
        // underpromotion
        int direction = -1;
        int promotion = -1;
        
        switch(chess::direction_of(move.from, move.to))
        {
        case chess::direction_nw:
            direction = 0;
            break;
        case chess::direction_n:
            direction = 1;
            break;
        case chess::direction_ne:
            direction = 2;
            break;
        default:
            throw std::logic_error("move action of invalid underpromotion direction");
        }

        switch(move.promote)
        {
        case chess::piece_rook:
            promotion = 0;
            break;
        case chess::piece_knight:
            promotion = 1;
            break;
        case chess::piece_bishop:
            promotion = 2;
            break;
        default:
            throw std::logic_error("move action of invalid underpromotion piece");
        }

        type = underpromotion_type + direction*underpromotion_pieces + promotion;
    }
    else if(piece == chess::piece_knight)
    {
        // knight
        int direction = -1;

        switch(chess::direction_of(move.from, move.to))
        {
        case chess::direction_nne:
            direction = 0;
            break;
        case chess::direction_ene:
            direction = 1;
            break;
        case chess::direction_ese:
            direction = 2;
            break;
        case chess::direction_sse:
            direction = 3;
            break;
        case chess::direction_ssw:
            direction = 4;
            break;
        case chess::direction_wsw:
            direction = 5;
            break;
        case chess::direction_wnw:
            direction = 6;
            break;
        case chess::direction_nnw:
            direction = 7;
            break;
        default:
            throw std::logic_error("move action of invalid knight direction " + std::to_string(chess::direction_of(move.from, move.to)));
        }

        type = knight_type + direction;
    }
    else
    {
        // sliding
        int direction = -1;
        int magnitude = -1;
        
        int steps = 0;
        int file_delta = chess::file_of(move.from) != chess::file_of(move.to);
        int rank_delta = chess::rank_of(move.from) != chess::rank_of(move.to);

        if(file_delta != 0)
        {
            steps = std::abs(file_delta);
        }
        else if(rank_delta != 0)
        {
            steps = std::abs(rank_delta);
        }
        else
        {
            throw std::logic_error("move action of sliding move with magnitude zero");
        }

        magnitude = steps - 1;

        chess::square step = chess::cat_coords
        (
            static_cast<chess::file>(chess::file_of(move.from) + file_delta/steps),
            static_cast<chess::rank>(chess::rank_of(move.from) + rank_delta/steps)
        );

        switch(chess::direction_of(move.from, step))
        {

        case chess::direction_n:
            direction = 0;
            break;
        case chess::direction_e:
            direction = 1;
            break;
        case chess::direction_s:
            direction = 2;
            break;
        case chess::direction_w:
            direction = 3;
            break;
        case chess::direction_ne:
            direction = 4;
            break;
        case chess::direction_se:
            direction = 5;
            break;
        case chess::direction_sw:
            direction = 6;
            break;
        case chess::direction_nw:
            direction = 7;
            break;
        default:
            throw std::logic_error("move action of invalid sliding direction");
        }

        type = sliding_type + direction*sliding_magnitudes + magnitude;
    }

    return square*actions_per_square + type;
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

    chess::side p1 = game.get_position().get_turn();
    chess::side p2 = chess::opponent(p1);

    const int planes = feature_planes*history + constant_planes;
    torch::Tensor input = torch::zeros({planes, 8, 8});

    int j = 0;

    // feature planes
    int n = std::min(history, static_cast<int>(game.size()+1));
    bool flip = p1 == chess::side_black;
    chess::game g = game;

    for(int i = 0; i < n; i++)
    {
        const chess::position& p = g.get_position();
        const chess::board& b = p.get_board();

        // p1 pieces
        for(int p = chess::piece_pawn; p <= chess::piece_king; p++)
        {
            chess::bitboard bb = b.piece_set(static_cast<chess::piece>(p), p1);
            torch::Tensor plane = bitboard_plane(bb);
            if(flip) plane = torch::flipud(plane);

            input.index_put_({j++}, plane);
        }

        // p2 pieces
        for(int p = chess::piece_pawn; p <= chess::piece_king; p++)
        {
            chess::bitboard bb = b.piece_set(static_cast<chess::piece>(p), p2);
            torch::Tensor plane = bitboard_plane(bb);
            if(flip) plane = torch::flipud(plane);

            input.index_put_({j++}, plane);
        }

        // repetitions
        int repetitions = g.get_repetitions();

        if(repetitions >= 1)
        {
            input.index_put_({j}, torch::ones({8, 8}));
        } j++;

        if(repetitions >= 2)
        {
            input.index_put_({j}, torch::ones({8, 8}));
        } j++;

        // previous position (if not at initial, in which case the loop will end)
        if(!g.empty())
        {
            g.pop();
        }
    }

    j = feature_planes*history;

    // constant planes
    const chess::position& position = game.get_position();

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

sigmanet make_network(int history, int filters, int blocks)
{
    return sigmanet(feature_planes*history + constant_planes, filters, blocks);
}


float material_value(const chess::game& game, chess::side side)
{
	const chess::board& board = game.get_position().get_board();

	float p1_value = 0.0f;
	float p2_value = 0.0f;

	for(int p = chess::piece_pawn; p < chess::piece_king; p++)
	{
		chess::piece piece = static_cast<chess::piece>(p);
		p1_value += chess::value_of(piece) * chess::set_cardinality(board.piece_set(piece, side));
		p2_value += chess::value_of(piece) * chess::set_cardinality(board.piece_set(piece, chess::opponent(side)));
	}

	if(p1_value + p2_value == 0.0f)
	{
		return 0.0f;
	}
	else
	{
		return (p1_value - p2_value) / (p1_value + p2_value);
	}
}


float material_delta(const chess::game& game, chess::side side)
{
	const chess::board& board = game.get_position().get_board();

	float p1_value = 0.0f;
	float p2_value = 0.0f;

	for(int p = chess::piece_pawn; p < chess::piece_king; p++)
	{
		chess::piece piece = static_cast<chess::piece>(p);
		p1_value += chess::value_of(piece) * chess::set_cardinality(board.piece_set(piece, side));
		p2_value += chess::value_of(piece) * chess::set_cardinality(board.piece_set(piece, chess::opponent(side)));
	}

    float value = p1_value - p2_value;

    if(value < 0.0f)
    {
        return -1.0f;
    }
    else if(value > 0.0f)
    {
        return 1.0f;
    }
    else
    {
        return 0.0f;
    }
}