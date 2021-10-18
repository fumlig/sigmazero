#include "policy.hpp"

#include "misc.hpp"


namespace policy
{


chess::move RandomPolicy::operator() (chess::position state) {
    std::vector<chess::move> available_moves{state.moves()};
    return *random_element(std::begin(available_moves), std::end(available_moves), generator);
}

} // namespace policy