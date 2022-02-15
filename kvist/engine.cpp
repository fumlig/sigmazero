#include <random>
#include <vector>
#include <atomic>
#include <chrono>
#include <functional>
#include <limits>
#include <unordered_map>

#include <chess/chess.hpp>
#include <uci/uci.hpp>


namespace kvist
{


struct entry
{
	chess::move pv;
};


std::unordered_map<std::size_t, entry> table;


float value(const chess::game& game)
{
	const chess::position& position = game.get_position();
	const chess::board& board = position.get_board();
	chess::side turn = position.get_turn();

	std::optional terminal_value = game.get_value(turn);

	if(terminal_value)
	{
		if(*terminal_value < 0)			return -std::numeric_limits<float>::infinity();
		else if(*terminal_value > 0)	return std::numeric_limits<float>::infinity();
		else							return 0.0f;
	}

	float material_value = 0.0f;

	for(int p = chess::piece_pawn; p < chess::piece_king; p++)
	{
		chess::piece piece = static_cast<chess::piece>(p);

		material_value += chess::value_of(piece) * chess::set_cardinality(board.piece_set(piece, turn));
		material_value -= chess::value_of(piece) * chess::set_cardinality(board.piece_set(piece, chess::opponent(turn)));
	}

	return material_value;
}


float quiesce(chess::game& game, float alpha, float beta)
{
	float v = value(game);

	if(v >= beta)
	{
		return beta;
	}

	if(alpha < v)
	{
		alpha = v;
	}

	const chess::position& position = game.get_position();
	const chess::board& board = position.get_board();

	for(chess::move move: position.moves())
	{
		auto [_, piece] = board.get(move.to);
		if(piece == chess::piece_none)
		{
			continue;
		}

		game.push(move);
		float value = -quiesce(game, -beta, -alpha);
		game.pop();

		if(value >= beta)
		{
			return beta;
		}

		if(value > alpha)
		{
			alpha = value;
		}
	}

	return alpha;
}


std::optional<float> search(chess::game& game, std::function<bool()> stop, int depth, float alpha = -std::numeric_limits<float>::infinity(), float beta = std::numeric_limits<float>::infinity())
{
	if(stop())
	{
		return std::nullopt;
	}

	if(depth == 0 || game.is_terminal())
	{
		return quiesce(game, alpha, beta);
	}

	bool nopv = true;
	float value = -std::numeric_limits<float>::infinity();

	for(chess::move move: game.get_position().moves())
	{
		game.push(move);

		std::optional<float> v;

		if(nopv)
		{
			v = search(game, stop, depth-1, -beta, -alpha);
		}
		else
		{
			v = search(game, stop, depth-1, -alpha-1, -alpha);
			
			if(v && -(*v) > alpha)
			{
				v = search(game, stop, depth-1, -beta, -alpha);
			}
		}

		game.pop();

		if(v)
		{
			value = -(*v);
		}
		else
		{
			return v;
		}

		if(value >= beta)
		{
			return beta;
		}

		if(value > alpha)
		{
			alpha = value;
			nopv = false;
		}
	}

	return alpha;
}

}


class engine : public uci::engine
{
private:
	chess::game game;

public:
	engine():
	game{}
	{

	}

	std::string name() const override
	{
		return "Kvist";
	}

	std::string author() const override
	{
		return "Oskar";
	}

	void setup(const chess::position& position, const std::vector<chess::move>& moves) override
	{
		game = chess::game(position, moves);
	}

	uci::search_result search(const uci::search_limit& limit, uci::search_info& info, const std::atomic_bool& ponder, const std::atomic_bool& stop) override
	{
        auto start_time = std::chrono::steady_clock::now();

        chess::side turn = game.get_position().get_turn();
        float clock = limit.clocks[turn];
        float increment = limit.increments[turn];

        int ply = game.size();
        int remaining_halfmoves = 59.3 + (72830 - 2330*ply)/(2644 + ply*(10 + ply));
        float budgeted_time = clock/remaining_halfmoves; // todo: increment

        auto stop_search = [&]()
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
                if(elapsed_time > limit.time || elapsed_time > budgeted_time)
                {
                    return true;
                }
            }

            return false;
        };


		std::vector<std::pair<chess::move, float>> moves;
		for(chess::move move: game.get_position().moves())
		{
			moves.emplace_back(move, -std::numeric_limits<float>::infinity());
		}

		int depth = 1;
		uci::search_result result = {moves.front().first};

		while(!stop_search())
		{
			bool nopv = true;
			float alpha = -std::numeric_limits<float>::infinity();
			float beta = std::numeric_limits<float>::infinity();

			for(auto& [move, value]: moves)
			{
				game.push(move);

				std::optional<float> v;

				if(nopv)
				{
					v = kvist::search(game, stop_search, depth-1, -beta, -alpha);
				}
				else
				{
					v = kvist::search(game, stop_search, depth-1, -alpha-1, -alpha);
					
					if(v && -(*v) > alpha)
					{
						v = kvist::search(game, stop_search, depth-1, -beta, -alpha);
					}
				}

				game.pop();

				if(v)
				{
					value = -(*v);
				}
				else
				{
					info.message("time spent");
					return result;
				}

				if(value >= beta)
				{
					break;
				}

				if(value > alpha)
				{
					alpha = value;
					nopv = false;
				}
			}

			std::sort(moves.begin(), moves.end(), [](auto m1, auto m2){return m1.second > m2.second;});
		
			auto [best_move, best_value] = moves.front();

			info.depth(depth);
			info.line({best_move});

			if(std::isinf(best_value))
			{
				// todo: do properly, use search depth
				if(best_value < 0)	info.mate(-1);
				if(best_value > 0)	info.mate(1);
			}
			else
			{
				info.score(best_value);
			}

			depth++;
			result.best = best_move;
		}

        return result;
	}

	void reset() override
	{

	}
};


int main(int argc, char** argv)
{
	chess::init();
	engine kvist;

	return uci::main(kvist);
}
