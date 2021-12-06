#include "utility.hpp"


std::mt19937& get_generator()
{
    static std::mt19937 generator = std::mt19937(std::random_device{}());
    return generator;
}


std::ostream& log(std::string_view type)
{
    return std::cerr << type << ": ";
}
