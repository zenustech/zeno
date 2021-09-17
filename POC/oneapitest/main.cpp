#include <CL/sycl.hpp>
#include <iostream>
#include <iomanip>
#include <cstdlib>

struct Dim3 {
    size_t x, y, z;

    Dim3(size_t x = 1, size_t y = 1, size_t z = 1)
        : x(x), y(y), z(z)
    {}
};

struct SyclQueue {
    sycl::queue m_q;

    void print_device_info() {
        auto device = m_q.get_device();
        auto p_name = device.get_platform().get_info<sycl::info::platform::name>();
        std::cout << std::setw(20) << "Platform Name: " << p_name << "\n";
        auto p_version = device.get_platform().get_info<sycl::info::platform::version>();
        std::cout << std::setw(20) << "Platform Version: " << p_version << "\n";
        auto d_name = device.get_info<sycl::info::device::name>();
        std::cout << std::setw(20) << "Device Name: " << d_name << "\n";
        auto max_work_group = device.get_info<sycl::info::device::max_work_group_size>();
        std::cout << std::setw(20) << "Max Work Group: " << max_work_group << "\n";
        auto max_compute_units = device.get_info<sycl::info::device::max_compute_units>();
        std::cout << std::setw(20) << "Max Compute Units: " << max_compute_units << "\n\n";
    }

    template <class Kernel>
    void parallel_for(Dim3 dim, Kernel kernel) {
        m_q.parallel_for(sycl::range<3>(dim.x, dim.y, dim.z), [=] (sycl::id<3> idx) {
            kernel(Dim3(idx[0], idx[1], idx[2]));
        }).wait();
    }

    void *malloc_device(size_t n) {
        return (void *)sycl::malloc_device<unsigned char>(n, m_q);
    }

    void *malloc_shared(size_t n) {
        return (void *)sycl::malloc_shared<unsigned char>(n, m_q);
    }

    void memcpy(void *d1, void *d2, size_t size) {
        m_q.memcpy(d1, d2, size).wait();
    }

    void memcpy_dtoh(void *h, void *d, size_t size) {
        m_q.memcpy(h, d, size).wait();
    }

    void memcpy_htod(void *h, void *d, size_t size) {
        m_q.memcpy(d, h, size).wait();
    }

    void free(void *p) {
        return sycl::free(p, m_q);
    }

    struct SharedAllocator {
        SyclQueue &m_q;

        SharedAllocator(SyclQueue &q) : m_q(q) {}

        void *allocate(size_t n) {
            return m_q.malloc_shared(n);
        }

        void *reallocate(void *old_p, size_t old_n, size_t new_n) {
            void *new_p = allocate(new_n);
            if (old_n) m_q.memcpy(new_p, old_p, std::min(old_n, new_n));
            deallocate(old_p);
            return new_p;
        }

        void deallocate(void *p) {
            m_q.free(p);
        }
    };

    SharedAllocator shared_allocator() {
        return {*this};
    }

    struct DeviceAllocator {
        SyclQueue &m_q;

        DeviceAllocator(SyclQueue &q) : m_q(q) {}

        void *allocate(size_t n) {
            return m_q.malloc_device(n);
        }

        void *reallocate(void *old_p, size_t old_n, size_t new_n) {
            void *new_p = allocate(new_n);
            if (old_n) m_q.memcpy(new_p, old_p, std::min(old_n, new_n));
            deallocate(old_p);
            return new_p;
        }

        void deallocate(void *p) {
            m_q.free(p);
        }
    };

    DeviceAllocator device_allocator() {
        return {*this};
    }
};

struct HostAllocator {
    void *allocate(size_t n) {
        return malloc(n);
    }

    void *zeroallocate(size_t n) {
        return calloc(n, 1);
    }

    void *reallocate(void *old_p, size_t old_n, size_t new_n) {
        return realloc(old_p, new_n);
    }

    void deallocate(void *p) {
        free(p);
    }
};

template <class T, class Alloc>
struct Vector {
    T *m_base{nullptr};
    size_t m_cap{0};
    size_t m_size{0};
    Alloc m_alloc;

    Vector(Alloc alloc, size_t n = 0)
        : m_base(n ? (T *)alloc.allocate(n * sizeof(T)) : nullptr)
        , m_cap(n), m_size(n)
        , m_alloc(std::move(alloc))
    {}

    ~Vector() {
        if (m_base) {
            m_alloc.deallocate(m_base);
            m_base = nullptr;
            m_cap = 0;
            m_size = 0;
        }
    }

    size_t size() const {
        return m_size;
    }

    size_t capacity() const {
        return m_cap;
    }

    void reserve(size_t n) {
        if (m_cap < n) {
            if (m_base) {
                m_base = (T *)m_alloc.reallocate(m_base, m_size * sizeof(T), n * sizeof(T));
            } else {
                m_base = (T *)m_alloc.allocate(m_base, n * sizeof(T));
            }
            m_cap = n;
        }
    }

    void shrink_to_fit() {
        if (m_cap > m_size) {
            if (m_size == 0) {
                m_alloc.deallocate(m_base);
                m_cap = 0;
                m_base = nullptr;
            } else {
                if (m_base) {
                    m_base = (T *)m_alloc.reallocate(m_base, m_size * sizeof(T), m_size * sizeof(T));
                } else {
                    m_base = (T *)m_alloc.allocate(m_base, m_size * sizeof(T));
                }
                m_cap = m_size;
            }
        }
    }

    void resize(size_t n) {
        reserve(n);
        m_size = n;
    }

    void clear() {
        resize(0);
    }

    T &operator[](size_t i) {
        return m_base[i];
    }

    T const &operator[](size_t i) const {
        return m_base[i];
    }

    using iterator = T *;
    using const_iterator = T const *;

    iterator begin() {
        return m_base;
    }

    iterator end() {
        return m_base + m_size;
    }

    const_iterator begin() const {
        return m_base;
    }

    const_iterator end() const {
        return m_base + m_size;
    }

    iterator __push_back() {
        size_t idx = m_size;
        size_t new_size = m_size + 1;
        if (new_size >= m_cap) {
            reserve(new_size + (new_size >> 1) + 1);
        }
        m_size = new_size;
        return m_base + idx;
    }

    void push_back(T const &val) {
        *__push_back() = val;
    }
};

int main(void) {
    SyclQueue q;
    Vector<int, SyclQueue::SharedAllocator> v(q.shared_allocator(), 32);
    q.parallel_for(Dim3(32, 1, 1), [=](Dim3 idx) {
        v[idx.x] = idx.x;
    });
    for (int i = 0; i < 32; i++) {
        printf("%d\n", v[i]);
    }
    return 0;
}
