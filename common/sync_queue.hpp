#ifndef SYNC_QUEUE_HPP
#define SYNC_QUEUE_HPP

#include <thread>
#include <chrono>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <atomic>


/**
 * Thread-safe queue class
 */
template <typename T>
class sync_queue {

    std::queue<T> container;
    std::mutex queue_lock;
    std::condition_variable queue_condition;
    std::atomic_int queue_size;
    static constexpr std::atomic_bool always_running = true;

public:

    // 
    const std::atomic_bool* running_ref = &always_running;

    /**
     * Returns the size of the queue
     */
    int size()
    {
        return queue_size;
    }

    /**
     * Synchronously pops an element from the queue
     */
    T pop()
    {

        std::unique_lock<std::mutex> lock(queue_lock);

        bool condition = true;
        while (condition)
        {
            auto duration = std::chrono::milliseconds(100);
            auto pred = [this]{ return !(*this->running_ref) || this->container.size() > 0; };

            queue_condition.wait_for(lock, duration, pred);

            if (size() || !*running_ref) condition = false;
        }

        // If the program is no longer running, but the queue is empty
        // return default constructed object TODO: maybe change this?
        if (container.empty())
        {
            lock.unlock();
            return T();
        }

        T result = container.front();

        queue_size--;
        container.pop();
        lock.unlock();

        return result;
    }

    /**
     * Synchronously pushes an element to the queue
     */
    void push(const T& data)
    {
        std::unique_lock<std::mutex> lock(queue_lock);

        container.push(data);
        queue_size++;

        lock.unlock();
        queue_condition.notify_one();
    }
};

#endif