#include <CL/sycl.hpp>
#include <iostream>
#include <memory>
#include <array>
#include "virtual_ptr.hpp"

namespace sycl = cl::sycl;

struct SyclSession {
    sycl::queue m_queue;
    sycl::codeplay::PointerMapper m_mapper;

    template <class F>
    decltype(auto) submit(F const &f) {
        return m_queue.submit(f);
    }

    decltype(auto) submit() {
        return m_queue.wait();
    }

    void *allocate(size_t size) {
        sycl::codeplay::SYCLmalloc(size, m_mapper);
    }

    void deallocate(void *ptr) {
        sycl::codeplay::SYCLfree(ptr, m_mapper);
    }

    auto &queue() {
        return m_queue;
    }

    auto &mapper() {
        return m_mapper;
    }
};


static SyclSession *syclSession() {
    static std::unique_ptr<SyclSession> g_session = std::make_unique<SyclSession>();
    return g_session.get();
}


template <class T>
struct SyclAllocator {
    using value_type = T;
    using pointer = T *;
    using const_pointer = const T *;

    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template <class U>
    struct rebind {
        using other = SyclAllocator<U>;
    };

    SyclAllocator() = default;
    ~SyclAllocator() = default;

    pointer allocate(size_type numObjects) {
        return static_cast<pointer>(syclSession()->allocate(sizeof(T) * numObjects));
    }

    pointer allocate(size_type numObjects, const void *hint) {
        return allocate(numObjects);
    }

    void deallocate(pointer p, size_type numObjects) {
        syclSession()->deallocate(p);
    }

    size_type max_size() const {
        return std::numeric_limits<size_type>::max();
    }

    template <class U, class... Args>
    void construct(U *p, Args &&... args) {
        new(p) U(std::forward<Args>(args)...);
    }

    template <class U>
    void destroy(U *p) {
        p->~U();
    }
};

class kernel0;

int main() {
    auto *sess = syclSession();

    SyclAllocator<int> svm;
    int *arr = svm.allocate(32);

    sess->queue().submit([&] (sycl::handler &cgh) {
        auto axr = sess->mapper().get_access<
            sycl::access::mode::read_write, sycl::access::target::global_buffer,
            int>(arr, cgh);
        cgh.parallel_for<kernel0>(sycl::range<1>(32), [=](sycl::item<1> id) {
            axr[id[0]] = id[0];
        });
    });

    {
        auto axr = sess->mapper().get_access<
            sycl::access::mode::read_write, sycl::access::target::host_buffer,
            int>(arr);
        for (int i = 0; i < 32; i++) {
            printf("%d\n", axr[i]);
        }
    }

    return 0;
}
