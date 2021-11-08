#ifndef MISC_H
#define MISC_H

#include <random>
#include <iterator>
#include <chrono>
#include <unordered_map>
#include <algorithm>

// Retrieve an iterator at a random position
template<typename Iterator, typename RandomGenerator>
Iterator random_element(Iterator start, Iterator end, RandomGenerator& generator);

// Retrieve a random element between two iterators
template<typename Iterator>
Iterator random_element(Iterator start, Iterator end);

// Get the index of the maximum element between two iterators
template<typename Iterator>
size_t get_max_idx(Iterator start, Iterator end);

// Get the max element between two iterators
template<typename T, typename Iterator1, typename Iterator2>
T get_max_element(Iterator1 wanted_start, Iterator2 comparator_start, Iterator2 comparator_end);

// Simple timer class for measuring execution time
struct Timer
{
    Timer();
    void set_start();
    double get_time(bool setstart=false);
    std::chrono::time_point<std::chrono::high_resolution_clock> start_t = std::chrono::high_resolution_clock::now();
};

// Simple function to get key(string) val(int) pairs from file to Map
std::unordered_map<std::string, int> parse_config(std::string filename);

#endif /* MISC_H */