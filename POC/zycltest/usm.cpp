#include <CL/sycl.hpp>


auto &default_queue() {
    static sycl::queue q;
    return q;
}


template <class T, sycl::usm::alloc alloc = sycl::usm::alloc::shared, size_t align = alignof(T)>
struct allocator : sycl::usm_allocator<T, alloc, align> {
    using sycl::usm_allocator<T, alloc, align>::usm_allocator;

    allocator() : allocator(default_queue()) {}
};

template <class T>
concept has_span = requires (T t) {
    *t.data();
    t.size();
};

template <class _pointer_type, class _size_type>
struct span {
    using pointer_type = _pointer_type;
    using size_type = _size_type;

    pointer_type _M_data;
    size_type _M_size;

    constexpr span(has_span auto &&t) noexcept
        : span(t.data(), t.size())
    {}
    constexpr span(pointer_type data, size_type size) noexcept
        : _M_data(data), _M_size(size)
    {}

    constexpr decltype(auto) operator[](size_type index) const {
        return _M_data[index];
    }

    constexpr pointer_type data() const {
        return _M_data;
    }

    constexpr size_type size() const {
        return _M_size;
    }
};

span(has_span auto &&t) -> span<decltype(t.data()), decltype(t.size())>;


int main() {
    std::vector<int, allocator<int>> arr(32);
    std::vector<int, allocator<int>> sum(1);

    span v_arr = arr;
    default_queue().parallel_for(
        sycl::nd_range<1>(arr.size(), 4),
        sycl::reduction(sum.data(), 0, std::plus{}),
        [=] (sycl::nd_item<1> it, auto &sum) {
            sum += v_arr[it.get_global_id(0)];
        }
    ).wait();

    printf("%d\n", sum[0]);
    return 0;
}
