#ifndef OUTPUT_THREAD_HPP
#define OUTPUT_THREAD_HPP

#include <atomic>
#include <string_view>

/**
 * Queues a message to be printed by the output thread
 * 
 * @param message   The content of the message
 */
void push_message(std::string_view message, bool endline = true);

/**
 * Main function ran by the output thread
 */
void output_thread_main(const std::atomic_bool& running);

#endif