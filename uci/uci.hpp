#ifndef UCI_HPP
#define UCI_HPP


#include <string>
#include <vector>
#include <atomic>
#include <unordered_map>
#include <unordered_set>
#include <initializer_list>
#include <sstream>
#include <numeric>
#include <memory>
#include <optional>
#include <stdexcept>
#include <functional>
#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <algorithm>
#include <thread>

#include <chess/chess.hpp>


namespace uci
{


struct identity
{
	std::string name;
	std::string author;
};


enum class option_type
{
	check,
	spin,
	combo,
	button,
	string
};


struct option
{
    option(option_type type);
    virtual ~option() = default;

    virtual void set(const std::string& value) = 0;

    virtual void insert(std::ostream& out) const = 0;

    std::string to_string() const;

    friend std::ostream& operator<<(std::ostream& out, const option& option);

    option_type type;
};

std::ostream& operator<<(std::ostream& out, const option& option);


struct option_check: option
{
    option_check(bool default_value);
    ~option_check() = default;

    operator bool() const;

    void set(const std::string& new_value) override;

    void insert(std::ostream& out) const override;

    bool value;
};

struct option_spin: option
{
    option_spin(int default_value, int min, int max);
    ~option_spin() = default;

    operator int() const;

    void set(const std::string& new_value) override;

    void insert(std::ostream& out) const override;

    int value;
    int min;
    int max;
};

struct option_combo: option
{
    option_combo(std::string_view default_value, std::initializer_list<std::string> alternatives);

    ~option_combo() = default;

    operator std::string() const;

    void set(const std::string& new_value) override;

    void insert(std::ostream& out) const override;

    std::string value;
    std::unordered_set<std::string> alternatives;
};

struct option_button: option
{
    option_button(std::function<void()> callback);
    ~option_button() = default;

    void set(const std::string&) override;

    void insert(std::ostream& out) const override;

    std::function<void()> callback;
};

struct option_string: option
{
    option_string(std::string_view default_value);
    ~option_string() = default;

    operator std::string() const;

    void set(const std::string& new_value) override;

    void insert(std::ostream& out) const override;

    std::string value;
};


class options
{
public:
    // Add option. Arguments forwarded to option constructor.
    template<class T, class... Ts> //requires std::derived_from<T, option>
    void add(const std::string& name, Ts&&... args);

    // Get option of by name.
    template<class T> //requires std::derived_from<T, option>
    const T& get(const std::string& name);

    // Set option value.
    void set(const std::string& name, const std::string& value);

    friend std::ostream& operator<<(std::ostream& out, const options& options);

    std::string to_string() const;

private:
    std::string key(const std::string& name) const;

    std::unordered_map<std::string, std::shared_ptr<option>> holder;
};

std::ostream& operator<<(std::ostream& out, const options& options);



// Limit position search. By default, search is infinite and stopped manually.
struct search_limit
{
    // Restrict search to moves.
    std::vector<chess::move> moves;

    // Search for specific time (seconds).
    float time = std::numeric_limits<float>::infinity();

    // Limit search depth (ply/halfmoves).
    std::optional<int> depth;

    // Only search a limited number of nodes.
    std::optional<int> nodes;

    // Search for mate in certain number of moves, negative if engine is getting mated.
    std::optional<int> mate;

    // Remaining clock time per side (seconds).
    std::array<float, chess::sides> clocks = {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()};

    // Increment per move per side (seconds).
    std::array<float, chess::sides> increments = {0.0f, 0.0f};

    // Moves to go until next time control.
    std::optional<int> remaining_moves;

    // Search until stop flag set, even if mate is found.
    bool infinite = false;
};


class search_info
{
public:
    search_info();

    // Inform about current search depth.
    void depth(int depth, std::optional<int> selective = std::nullopt);

    // Inform about number of nodes searched.
    void nodes(unsigned long long nodes);

    // Best line.
    void line(const std::vector<chess::move>& best);

    // Evaluation score.
    void score(float centipawn);

    // New lower or upper bound found.
    void bounds(float lower, float upper);

    // Mate has been found. Negative value means engine is getting mated.
    void mate(int moves);

    // Current move being searched.
    void move(const chess::move& current, int number);

    // Send info message.
    void message(const std::string& message);

private:
    std::chrono::time_point<std::chrono::steady_clock> search_start;
    unsigned long long last_nodes;
    std::chrono::time_point<std::chrono::steady_clock> last_nodes_time;
};


struct search_result
{
    // Best move found by engine.
    chess::move best;

    // Move suspected to be played by opponent.
    std::optional<chess::move> ponder;
};


// Implement this interface to make an engine usable with UCI.
// Should be able to return best move after a search, and also the move it likes to ponder on.
class engine
{
public:
    virtual ~engine() = default;

    // Engine name.
    virtual std::string name() const = 0;

    // Engine author.
    virtual std::string author() const = 0;

    // Set up position.
    virtual void setup(const chess::position& position, const std::vector<chess::move>& moves) = 0;

    // Start calculating on current position. Info should be sent regularly. The search is to be stopped when
    // the stop flag is set, or when mate is found unless the search is infinite or ponder is on.
    // The stop and ponder flags may be set to false during the search. When the search is stopped,
    // the function should return the best move and what move is thought to be the best answer to the best move.
    virtual search_result search(const search_limit& limit, search_info& info, const std::atomic_bool& ponder, const std::atomic_bool& stop) = 0;

    // Called before a new position will be sent.
    virtual void reset() = 0;

    // Engine options.
    options opt;
};


// Call this with an engine to start communicating with UCI.
int main(engine& engine);


}


template<class T, class... Args> //requires std::derived_from<T, uci::option>
void uci::options::add(const std::string& name, Args&&... args)
{
    holder.emplace(key(name), std::make_unique<T>(std::forward<Args>(args)...));
}


template<class T> //requires std::derived_from<T, uci::option>
const T& uci::options::get(const std::string& name)
{
    return dynamic_cast<T&>(*holder.at(key(name)));
}


#endif
