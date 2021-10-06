#ifndef MISC_H
#define MISC_H

#include <random>
#include <iterator>
#include <chrono>
#include <fstream>
#include <unordered_map>
#include <string>
#include <sstream>
#include <algorithm>

// Retrieve an iterator at a random position
template<typename Iterator, typename RandomGenerator>
Iterator random_element(Iterator start, Iterator end, RandomGenerator& generator)
{
    std::uniform_int_distribution<> dist(0, std::distance(start, end)-1);
    std::advance(start, dist(generator));
    return start;
};

// Retrieve a random element between two iterators
template<typename Iterator>
Iterator random_element(Iterator start, Iterator end)
{
    static std::random_device random_device;
    static std::mt19937 generator(random_device());
    return random_element(start, end, generator);
};

// Get the index of the maximum element between two iterators
template<typename Iterator>
size_t get_max_idx(Iterator start, Iterator end)
{
    return std::distance(start, std::max_element(start, end));
};

// Get the max element between two iterators
template<typename T, typename Iterator1, typename Iterator2>
T get_max_element(Iterator1 wanted_start, Iterator2 comparator_start, Iterator2 comparator_end)
{
    size_t max_idx = get_max_idx(comparator_start, comparator_end);
    std::advance(wanted_start, max_idx);
    return *wanted_start;
};

// Simple timer class for measuring execution time
struct Timer
{
    Timer()
    {}
    void set_start()
    {
        start_t = std::chrono::high_resolution_clock::now();
    }
    double get_time(bool setstart=false)
    {
        auto new_t = std::chrono::high_resolution_clock::now();
        auto duration = new_t-start_t;
        double time = std::chrono::duration<double>(duration).count();
        if(setstart) set_start();
        return time;
    }

    std::chrono::time_point<std::chrono::high_resolution_clock> start_t = std::chrono::high_resolution_clock::now();
};

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

#endif /* MISC_H */