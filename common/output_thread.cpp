#include "output_thread.hpp"

#include <string>
#include <iostream>

#include "sync_queue.hpp"

struct message {
    std::string content;
    bool endline;
};

static sync_queue<message> message_queue;

void push_message(std::string_view message, bool endline)
{
    message_queue.push({std::string(message), endline});
}


void process_message()
{
    auto message = message_queue.pop();
    std::cout << message.content;
    if (message.endline) std::cout << std::endl;
}


void output_thread_main(const std::atomic_bool& running)
{
    message_queue.running_ref = &running;

    while (running || message_queue.size() > 0)
    {
        process_message();
    }
}
