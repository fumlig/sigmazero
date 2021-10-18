
#include "misc.hpp"
#include <chrono>
#include <fstream>
#include <unordered_map>
#include <string>
#include <sstream>


Timer::Timer(){}

void Timer::set_start()
{
    start_t = std::chrono::high_resolution_clock::now();
}

double Timer::get_time(bool setstart)
{
    auto new_t = std::chrono::high_resolution_clock::now();
    auto duration = new_t-start_t;
    double time = std::chrono::duration<double>(duration).count();
    if(setstart) set_start();
    return time;
}

// Simple function to get key(string) val(int) pairs from file to Map
std::unordered_map<std::string, int> parse_config(std::string filename)
{
    std::unordered_map<std::string, int> dict{};
    std::ifstream is(filename);
    std::string line;
    size_t sep_idx;
    while(std::getline(is, line))
    {
        sep_idx = line.find('=');
        std::string key{line.substr(0, sep_idx)};
        std::string val{line.substr(sep_idx+1)};
        dict.insert({key, std::stoi(val)});
    }
    return dict;
}

