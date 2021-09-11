#include <CL/sycl.hpp>
#include <iostream>
#include <memory>
#include <array>

namespace sycl = cl::sycl;


struct Handler {
    sycl::handler &m_handler;
    Handler(sycl::handler &handler) : m_handler(handler) {}

    template <class Key, class RangeT, class KernelT>
    void parallel_for(RangeT &&range, KernelT &&kernel) {
        m_handler.parallel_for<Key>(std::forward(range), [&] (auto const &id) {
            kernel(id);
        });
    }
};


struct Queue {
    sycl::queue m_queue;

    template <class Func>
    void enqueue(Func const &func) {
        m_queue.submit([&] (sycl::handler &cgh) {
            Handler handler(cgh);
            func(handler);
        });
    }

    void wait() {
        m_queue.wait();
    }
};

static Queue *getQueue() {
    std::unique_ptr<Queue> g_queue;
    return g_queue;
}


enum AccessorType {
    ReadOnly,
    WriteOnly,
    ReadWrite,
};


struct Accessor {
};


struct Buffer {
    sycl::buffer<char> m_buffer;

    Buffer(void *data, size_t size) : m_buffer((char *)data, size) {}

    template <AccessorType Type = AccessorType::ReadWrite>
    auto getAccessor() {
    }
};


template <typename T>
class SimpleVadd;

template <typename T, size_t N>
void simple_vadd(std::array<T, N> const &VA, std::array<T, N> const &VB,
        std::array<T, N> &VC) {
    Buffer bufA((void *)VA.data(), N * sizeof(T));
    Buffer bufB((void *)VB.data(), N * sizeof(T));
    Buffer bufC((void *)VC.data(), N * sizeof(T));

    getQueue().enqueue([&] (auto &hdl) {
        auto axrA = bufA.getAccessor<AccessorType::ReadOnly>();
        auto axrB = bufB.getAccessor<AccessorType::ReadOnly>();
        auto axrC = bufC.getAccessor<AccessorType::WriteOnly>();
        hdl.parallel_for<SimpleVadd<T>>(N, [=](auto id) {
            axrC[id] = axrA[id] + axrB[id];
        });
    });
}


int main() {
    constexpr size_t array_size = 4;
    std::array<int, array_size> A = {1, 2, 3, 4}, B = {1, 2, 3, 4}, C;
    simple_vadd(A, B, C);
    for (unsigned int i = 0; i < array_size; i++) {
        if (C[i] != A[i] + B[i]) {
            std::cout << "The results are incorrect (element " << i << " is " << C[i]
                << "!\n";
            return 1;
        }
    }
    std::cout << "The results are correct!\n";
    return 0;
}
