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
    if (!g_queue) g_queue = std::make_unique<Queue>();
    return g_queue.get();
}


enum AccessorType : int {
    ReadOnly = 1,
    WriteOnly = 2,
    ReadWrite = 3,
};

template <AccessorType Type>
static constexpr auto __sycl_accessor_type() {
    if constexpr (Type == AccessorType::ReadOnly) {
        return sycl::access::mode::read;
    } else if constexpr (Type == AccessorType::WriteOnly) {
        return sycl::access::mode::write;
    } else if constexpr (Type == AccessorType::ReadWrite) {
        return sycl::access::mode::read_write;
    } else {
        static_assert(Type != Type, "invalid AccessorType!");
    }
}


template <AccessorType Type>
struct Accessor {
    using __SyclAccessorType = sycl::accessor<char, 1, __sycl_accessor_type<Type>()>;
    __SyclAccessorType m_accessor;

    Accessor(__SyclAccessorType &&accessor) : m_accessor(std::move(accessor)) {
    }
};


struct Buffer {
    sycl::buffer<char, 1> m_buffer;

    Buffer(void *data, size_t size) : m_buffer((char *)data, size) {}

    template <AccessorType Type = AccessorType::ReadWrite>
    auto getAccessor() {
        return Accessor(m_buffer.get_access<__sycl_accessor_type<Type>()>());
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

    getQueue()->enqueue([&] (Handler &hdl) {
        auto axrA = bufA.getAccessor<AccessorType::ReadOnly>();
        auto axrB = bufB.getAccessor<AccessorType::ReadOnly>();
        auto axrC = bufC.getAccessor<AccessorType::WriteOnly>();
        hdl.parallel_for<SimpleVadd<T>>(N, [=](auto id) {
            axrC[id[0]] = axrA[id[0]] + axrB[id[0]];
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
