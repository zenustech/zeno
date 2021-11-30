/**
 * Copyright (c) 2020-2021 CutDigital Ltd.
 * All rights reserved.
 * 
 * NOTE: This file is licensed under GPL-3.0-or-later (default). 
 * A commercial license can be purchased from CutDigital Ltd. 
 *  
 * License details:
 * 
 * (A)  GNU General Public License ("GPL"); a copy of which you should have 
 *      recieved with this file.
 * 	    - see also: <http://www.gnu.org/licenses/>
 * (B)  Commercial license.
 *      - email: contact@cut-digital.com
 * 
 * The commercial license options is for users that wish to use MCUT in 
 * their products for comercial purposes but do not wish to release their 
 * software products under the GPL license. 
 * 
 * Author(s)     : Floyd M. Chitalu
 */

#ifndef MCUT_SCHEDULER_H_
#define MCUT_SCHEDULER_H_

#include <thread>
#include <condition_variable>
#include <mutex>
#include <atomic>
#include <future>
#include <memory>
#include <vector>

//#define USE_LOCKFREE_WORKQUEUE 1

namespace mcut
{

    class function_wrapper
    {
    private:
        struct impl_base
        {
            virtual void call() = 0;
            virtual ~impl_base() {}
        };

        std::unique_ptr<impl_base> impl;

        template <typename F>
        struct impl_type : impl_base
        {
            F f;
            impl_type(F &&f_) : f(std::move(f_)) {}
            void call() { return f(); }
        };

    public:
        template <typename F>
        function_wrapper(F &&f) : impl(new impl_type<F>(std::move(f)))
        {
        }

        void operator()() { impl->call(); }

        function_wrapper() = default;

        function_wrapper(function_wrapper &&other) : impl(std::move(other.impl))
        {
        }

        function_wrapper &operator=(function_wrapper &&other)
        {
            impl = std::move(other.impl);
            return *this;
        }

        function_wrapper(const function_wrapper &) = delete;
        function_wrapper(function_wrapper &) = delete;
        function_wrapper &operator=(const function_wrapper &) = delete;
    };

#if defined(USE_LOCKFREE_WORKQUEUE)
    template <typename T>
    class lock_free_queue
    {
    private:
        struct node;

        struct counted_node_ptr
        {
            int external_count = 0;
            node *ptr = nullptr;
        };
        std::atomic<counted_node_ptr> head;
        std::atomic<counted_node_ptr> tail;
        struct node_counter
        {
            unsigned internal_count : 30;
            unsigned external_counters : 2;
        };

        struct node
        {
            std::atomic<T *> data;
            std::atomic<node_counter> count;
            std::atomic<counted_node_ptr> next;

            node()
            {
                node_counter new_count;
                new_count.internal_count = 0;
                new_count.external_counters = 2;
                count.store(new_count);
                //next.ptr = nullptr;
                //next.external_count = 0;
            }

            void release_ref()
            {
                node_counter old_counter = count.load(std::memory_order_relaxed);
                node_counter new_counter;
                do
                {
                    new_counter = old_counter;
                    --new_counter.internal_count;
                } while (!count.compare_exchange_strong(
                    old_counter, new_counter,
                    std::memory_order_acquire, std::memory_order_relaxed));

                if (!new_counter.internal_count &&
                    !new_counter.external_counters)
                {
                    delete this;
                }
            }
        };

        static void increase_external_count(
            std::atomic<counted_node_ptr> &counter,
            counted_node_ptr &old_counter)
        {
            counted_node_ptr new_counter;
            do
            {
                new_counter = old_counter;
                ++new_counter.external_count;
            } while (!counter.compare_exchange_strong(
                old_counter, new_counter,
                std::memory_order_acquire, std::memory_order_relaxed));
            old_counter.external_count = new_counter.external_count;
        }

        static void free_external_counter(counted_node_ptr &old_node_ptr)
        {
            node *const ptr = old_node_ptr.ptr;
            int const count_increase = old_node_ptr.external_count - 2;
            node_counter old_counter =
                ptr->count.load(std::memory_order_relaxed);

            node_counter new_counter;
            do
            {
                new_counter = old_counter;
                --new_counter.external_counters;
                new_counter.internal_count += count_increase;
            } while (!ptr->count.compare_exchange_strong(
                old_counter, new_counter,
                std::memory_order_acquire, std::memory_order_relaxed));

            if (!new_counter.internal_count &&
                !new_counter.external_counters)
            {
                delete ptr;
            }
        }

        void set_new_tail(counted_node_ptr &old_tail,
                          counted_node_ptr const &new_tail)
        {
            node *const current_tail_ptr = old_tail.ptr;
            while (!tail.compare_exchange_weak(old_tail, new_tail) &&
                   old_tail.ptr == current_tail_ptr)
                ;
            if (old_tail.ptr == current_tail_ptr)
                free_external_counter(old_tail);
            else
                current_tail_ptr->release_ref();
        }

    public:
        lock_free_queue()
        {
            counted_node_ptr cnp;
            cnp.external_count = 0;
            cnp.ptr = new node;
            head.store(cnp, std::memory_order_seq_cst);
            tail.store(head.load(std::memory_order_seq_cst));
        }
        lock_free_queue(const lock_free_queue &other) = delete;
        lock_free_queue &operator=(const lock_free_queue &other) = delete;
        ~lock_free_queue()
        {
            counted_node_ptr old_head;
            while ((old_head = head.load()).ptr != nullptr)
            {
                head.store(old_head.ptr->next);
                delete old_head.ptr;
                old_head.ptr = nullptr;
            }
        }
        std::unique_ptr<T> pop()
        {
            counted_node_ptr old_head = head.load(std::memory_order_relaxed);
            for (;;)
            {
                increase_external_count(head, old_head);
                node *const ptr = old_head.ptr;
                if (ptr == tail.load().ptr)
                {
                    return std::unique_ptr<T>();
                }
                counted_node_ptr next = ptr->next.load();
                if (head.compare_exchange_strong(old_head, next))
                {
                    T *const res = ptr->data.exchange(nullptr);
                    free_external_counter(old_head);
                    return std::unique_ptr<T>(res);
                }
                ptr->release_ref();
            }
        }

        void push(T new_value)
        {
            std::unique_ptr<T> new_data(new T(std::move(new_value)));
            counted_node_ptr new_next;
            new_next.ptr = new node;
            new_next.external_count = 1;
            counted_node_ptr old_tail = tail.load();
            for (;;)
            {
                increase_external_count(tail, old_tail);
                T *old_data = nullptr;
                if (old_tail.ptr->data.compare_exchange_strong(
                        old_data, new_data.get()))
                {
                    counted_node_ptr old_next; // = {0, nullptr};
                    if (!old_tail.ptr->next.compare_exchange_strong(
                            old_next, new_next))
                    {
                        delete new_next.ptr;
                        new_next = old_next;
                    }
                    set_new_tail(old_tail, new_next);
                    new_data.release();
                    break;
                }
                else
                {
                    counted_node_ptr old_next; // = {0, nullptr};
                    if (old_tail.ptr->next.compare_exchange_strong(
                            old_next, new_next))
                    {
                        old_next = new_next;
                        new_next.ptr = new node;
                    }
                    set_new_tail(old_tail, old_next);
                }
            }
        }
    };
#else
    template <typename T>
    class thread_safe_queue
    {
    private:
        struct node
        {
            std::shared_ptr<T> data;
            std::unique_ptr<node> next;
        };

        std::mutex head_mutex;
        std::unique_ptr<node> head;
        std::mutex tail_mutex;
        node *tail;
        std::condition_variable data_cond;
        std::atomic_bool can_wait_for_data;

        std::unique_ptr<node> try_pop_head(T &value)
        {
            std::lock_guard<std::mutex> head_lock(head_mutex);
            if (head.get() == get_tail())
            {
                return std::unique_ptr<node>(nullptr);
            }
            value = std::move(*head->data);
            return pop_head();
        }

        node *get_tail()
        {
            std::lock_guard<std::mutex> tail_lock(tail_mutex);
            return tail;
        }

        // dont call directly
        std::unique_ptr<node> pop_head()
        {
            std::unique_ptr<node> old_head = std::move(head);
            head = std::move(old_head->next);
            return (old_head);
        }

        std::unique_lock<std::mutex> wait_for_data()
        {
            std::unique_lock<std::mutex> head_lock(head_mutex);
            auto until = [&]()
            { return !can_wait_for_data || head.get() != get_tail(); };
            data_cond.wait(head_lock, until);
            return head_lock;
        }

        std::unique_ptr<node> wait_pop_head(T &value)
        {
            std::unique_lock<std::mutex> head_lock(wait_for_data());
            if (can_wait_for_data)
            {
                value = std::move(*head->data);
                return pop_head();
            }
            else
            {
                return std::unique_ptr<node>(nullptr);
            }
        }

    public:
        thread_safe_queue() : head(new node), tail(head.get()), can_wait_for_data(true) {}
        thread_safe_queue(const thread_safe_queue &other) = delete;
        thread_safe_queue &operator=(const thread_safe_queue &other) = delete;

        void disrupt_wait_for_data()
        {
            can_wait_for_data = false;
            data_cond.notify_one();
        }

        void push(T new_value)
        {
            std::shared_ptr<T> new_data(std::make_shared<T>(std::move(new_value)));
            std::unique_ptr<node> p(new node);
            {
                std::lock_guard<std::mutex> tail_lock(tail_mutex);
                tail->data = new_data;
                node *const new_tail = p.get();
                tail->next = std::move(p);
                tail = new_tail;
            }
            data_cond.notify_one();
        }

        void wait_and_pop(T &value)
        {
            std::unique_ptr<node> const old_head = wait_pop_head(value);
        }

        bool try_pop(T &value)
        {
            std::unique_ptr<node> const old_head = try_pop_head(value);
            return !(!(old_head)); // https://stackoverflow.com/questions/30521849/error-on-implicit-cast-from-stdunique-ptr-to-bool
        }

        void empty()
        {
            std::lock_guard<std::mutex> head_lock(head_mutex);
            return (head.get() == get_tail());
        }
    };
#endif

    class join_threads
    {
        std::vector<std::thread> &threads;

    public:
        explicit join_threads(std::vector<std::thread> &threads_) : threads(threads_)
        {
        }
        ~join_threads()
        {
            for (unsigned long i = 0; i < threads.size(); ++i)
            {
                if (threads[i].joinable())
                    threads[i].join();
            }
        }
    };

    class thread_pool
    {
        std::atomic_bool terminate;
#if defined(USE_LOCKFREE_WORKQUEUE)
        std::atomic_bool busy_wait_enabled; // i.e. we are in the mcDispatch function
        std::mutex busy_wait_mutex;
        std::condition_variable busy_wait_cond;
#endif

#if defined(USE_LOCKFREE_WORKQUEUE)
        std::vector<lock_free_queue<function_wrapper>> work_queues;
#else
        std::vector<thread_safe_queue<function_wrapper>> work_queues;
#endif
        std::vector<std::thread> threads; // NOTE: must be declared after "terminate" and "work_queues"
        join_threads joiner;
        unsigned long long round_robin_scheduling_counter;

#if defined(USE_LOCKFREE_WORKQUEUE)
        std::unique_ptr<function_wrapper> pop_from_other_thread_queue(const int worker_thread_id)
        {
            std::unique_ptr<function_wrapper> stolen_task;
            const unsigned num_work_queues = (unsigned)work_queues.size();
            for (unsigned i = 0; i < num_work_queues; ++i)
            {
                unsigned const other_worker_thread_id = (worker_thread_id + i + 1) % num_work_queues;
                stolen_task = work_queues[other_worker_thread_id].pop();
                if (stolen_task.get() != nullptr)
                {
                    break;
                }
            }
            return stolen_task;
        }
#else
        bool try_pop_from_other_thread_queue(function_wrapper &task, const int worker_thread_id)
        {
            const unsigned num_work_queues = (unsigned)work_queues.size();
            for (unsigned i = 0; i < num_work_queues; ++i)
            {
                unsigned const other_worker_thread_id = (worker_thread_id + i + 1) % num_work_queues;
                if (work_queues[other_worker_thread_id].try_pop(task))
                {
                    return true;
                }
            }
            return false;
        }
#endif

#if defined(USE_LOCKFREE_WORKQUEUE)
        // The following functions are used to control when worker threads should
        // actively begin consuming CPU resources/cyles by busy-waiting for tasks.
        // For example, signalling to stop busy-waiting means that there is no more
        // work to do in-so-far-as mcDispatch is concerned. So in this instance,
        // workers should block until called upon in the future with another
        // mcDispatch API call from the client application.
        // We need to control this behaviour to prevent worker-thread from consuming CPU
        // time unnecessarily, especially since the client-application might need
        // those resourses!

        // wait (block) until the mcDispatch API function is called by
        // the user, or the program has been terminated.
        void client_wait()
        {
            std::unique_lock<std::mutex> lk(busy_wait_mutex);
            busy_wait_cond.wait(lk, [&]
                                { return !terminate && !busy_wait_enabled; });
        }

        // signal that all worker-threads should enter a busy-wait loop, for tasks
        // that are submitted during the mcDispatch API function
        // This is acceccible/called only by a busy_wait_guard object
        void start_busy_wait()
        {
            busy_wait_enabled = true;
        }

        // signal that all worker-threads should exit the busy-wait loop, for tasks
        // that are submitted during the mcDispatch API function.
        // This is acceccible/called only by a busy_wait_guard object
        void stop_busy_wait()
        {
            busy_wait_enabled = false;
        }
#endif

        void worker_thread(int worker_thread_id)
        {
#if defined(USE_LOCKFREE_WORKQUEUE)
            while (true) // infinite loop lasting the lifetime of an MCUT context
            {
                client_wait();

                while (busy_wait_enabled) // loop lasting the lifetime of an MCUT mcDispatch API function call
                {
                    std::unique_ptr<function_wrapper> task;

                    // try to pop some work to do from my queue, or steal from some other worker-thread's queue.
                    if ((task = work_queues[worker_thread_id].pop()).get() != nullptr /*|| //
                        (task = pop_from_other_thread_queue(worker_thread_id)).get() != nullptr*/
                    )
                    {
                        (*task)();
                    }

                    if (terminate)
                    {
                        break; // finished (i.e. MCUT context was destroyed)
                    }
                }

                if (terminate)
                {
                    break; // finished (i.e. MCUT context was destroyed)
                }
            } //
#else         // #if defined(USE_LOCKFREE_WORKQUEUE)

            do
            {
                function_wrapper task;
#if 0
                work_queues[worker_thread_id].wait_and_pop(task);
                if(terminate) {
                   break; // finished (i.e. MCUT context was destroyed)
                }
                task();
#else

                // if I can't pop any task from my queue, and I can't steal a task from
                // another thread's queue, then I'll just wait until is added to my queue.
                if (!(work_queues[worker_thread_id].try_pop(task) || try_pop_from_other_thread_queue(task, worker_thread_id)))
                {
                    work_queues[worker_thread_id].wait_and_pop(task);
                }

                if (terminate)
                {
                    break; // finished (i.e. MCUT context was destroyed)
                }

                task(); // run the task
#endif
            } while (true);
#endif //#if defined(USE_LOCKFREE_WORKQUEUE)
        }

    public:
#if defined(USE_LOCKFREE_WORKQUEUE)
        /*
            The class busy_wait_guard is a mini wrapper that provides a convenient RAII-style 
            mechanism for enabling a threadpool's "busy-wait" state for the duration of a 
            "mcDispatch" API function call.

            When a busy_wait_guard object is created, it starts the busy-wait state for all 
            worker threads (of the current context), in which these workers actively query for 
            tasks to do. When control leaves the mcDispatch API function scope in which the 
            busy_wait_guard object was created, the busy_wait_guard is destructed and the 
            busy-wait state is stopped, resulting in all worker-threads waiting until 1) there 
            is another mcDispatch API function call from the client/user applcation, or 2) the 
            user/client application destroys the MCUT context object associated with the 
            respective thread pool, which destroys the thread-pool.

            The lock_guard class is non-copyable. 
        */
        class busy_wait_guard
        {
            friend class thread_pool;
            thread_pool *const pool;

        public:
            busy_wait_guard(thread_pool *pool_) : pool(pool_)
            {
                pool->start_busy_wait();
            }
            ~busy_wait_guard()
            {
                pool->stop_busy_wait();
            }

        protected:
            busy_wait_guard() : pool(nullptr) {}

        private: // emphasize the following members are private
            busy_wait_guard(const busy_wait_guard &);
            const busy_wait_guard &operator=(const busy_wait_guard &);
        };
#endif // #if defined(USE_LOCKFREE_WORKQUEUE)

        thread_pool() : terminate(false),
#if defined(USE_LOCKFREE_WORKQUEUE)
                        busy_wait_enabled(false),
#endif
                        joiner(threads), round_robin_scheduling_counter(0)
        {
            unsigned int const thread_count = std::thread::hardware_concurrency();

            try
            {
#if defined(USE_LOCKFREE_WORKQUEUE)
                work_queues = std::vector<lock_free_queue<function_wrapper>>(thread_count);
#else
                work_queues = std::vector<thread_safe_queue<function_wrapper>>(
                    thread_count);
#endif
                for (unsigned i = 0; i < thread_count; ++i)
                {

                    threads.push_back(std::thread(&thread_pool::worker_thread, this, i));
                }
            }
            catch (...)
            {
                terminate = true;
                wakeup_and_shutdown();
                throw;
            }
        }

        ~thread_pool()
        {
#if defined(USE_LOCKFREE_WORKQUEUE)
            stop_busy_wait();
#endif // #if defined(USE_LOCKFREE_WORKQUEUE)

            terminate = true;

#if defined(USE_LOCKFREE_WORKQUEUE)
#else  // #if defined(USE_LOCKFREE_WORKQUEUE)

            wakeup_and_shutdown();
#endif // #if defined(USE_LOCKFREE_WORKQUEUE)
        }

        // submit empty task so that worker threads can wake up
        // with a valid (but redundant) task to then exit
        void wakeup_and_shutdown()
        {
            for (unsigned i = 0; i < get_num_threads(); ++i)
            {
                work_queues[i].disrupt_wait_for_data();
            }
        }

    public:
        /*
        The thread pool takes care of the exception safety too. Any exception thrown by the
        task gets propagated through the std::future returned from submit() , and if the function
        exits with an exception, the thread pool destructor abandons any not-yet-completed
        tasks and waits for the pool threads to finish.
    */
        template <typename FunctionType>
        std::future<typename std::result_of<FunctionType()>::type> submit(FunctionType f)
        {
            typedef typename std::result_of<FunctionType()>::type result_type;

            std::packaged_task<result_type()> task(std::move(f));
            std::future<result_type> res(task.get_future());

            unsigned long long worker_thread_id = (round_robin_scheduling_counter++) % (unsigned long long)get_num_threads();

            //printf("[MCUT]: submit to thread %d\n", (int)worker_thread_id);

            work_queues[worker_thread_id].push(std::move(task));

            return res;
        }

        size_t get_num_threads() const
        {
            return threads.size();
        }
    };

    template <typename InputStorageIteratorType, typename OutputStorageType, typename FunctionType>
    void parallel_fork_and_join(
        thread_pool &pool,
        // start of data elements to be processed in parallel
        const InputStorageIteratorType &first,
        // end of of data elements to be processed in parallel (e.g. std::map::end())
        const InputStorageIteratorType &last,
        // the ideal size of the block assigned to each thread
        typename InputStorageIteratorType::difference_type const block_size_default,
        // the function that is executed on a sub-block of element within the range [first, last)
        FunctionType &task_func,
        // the part of the result/output that is computed by the master thread (i.e. the one that is scheduling)
        // NOTE: this result must be merged which the output computed for the other
        // sub-block in the input ranges. This other data is accessed from the std::futures
        OutputStorageType &master_thread_output,
        // Future promises of data (to be merged) that is computed by worker threads
        std::vector<std::future<OutputStorageType>> &futures)
    {

        typename InputStorageIteratorType::difference_type const length = std::distance(first, last);
        typename InputStorageIteratorType::difference_type const block_size = std::min(block_size_default, length);
        typename InputStorageIteratorType::difference_type const num_blocks = (length + block_size - 1) / block_size;

        //std::cout << "length=" << length << " block_size=" << block_size << " num_blocks=" << num_blocks << std::endl;

        futures.resize(num_blocks - 1);
        InputStorageIteratorType block_start = first;

        for (typename InputStorageIteratorType::difference_type i = 0; i < (num_blocks - 1); ++i)
        {
            InputStorageIteratorType block_end = block_start;
            std::advance(block_end, block_size);

            futures[i] = pool.submit(
                [&, block_start, block_end]() -> OutputStorageType
                {
                    return task_func(block_start, block_end);
                });

            block_start = block_end;
        }

        master_thread_output = task_func(block_start, last);
    }

} // namespace mcut{

#endif // MCUT_SCHEDULER_H_