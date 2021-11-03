#include <CL/sycl.hpp>


namespace zpc {


using f32 = float;
using f64 = double;
using isize = std::intptr_t;
using usize = std::uintptr_t;
using i8 = std::int8_t;
using u8 = std::uint8_t;
using i16 = std::int16_t;
using u16 = std::uint16_t;
using i32 = std::int32_t;
using u32 = std::uint32_t;
using i64 = std::int64_t;
using u64 = std::uint64_t;


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


auto parallel_for(auto &&...args) {
    return default_queue().parallel_for(std::forward<decltype(args)>(args)...);
}

auto single_task(auto &&...args) {
    return default_queue().single_task(std::forward<decltype(args)>(args)...);
}

auto submit(auto &&...args) {
    return default_queue().submit(std::forward<decltype(args)>(args)...);
}

auto copy(auto &&...args) {
    return default_queue().copy(std::forward<decltype(args)>(args)...);
}


template <class T>
using vector = std::vector<T, allocator<T>>;


}


int main() {
    zpc::vector<float> arr(4096);
    zpc::vector<float> sum(1);

    for (auto &a: arr) {
        a = drand48();
    }

    zpc::span v_arr = arr;
    zpc::parallel_for(
        sycl::nd_range<1>(arr.size(), 256),
        sycl::reduction(sum.data(), 0.0f, std::plus{}),
        [=] (sycl::nd_item<1> it, auto &sum) {
            sum += v_arr[it.get_global_id(0)];
        }
    ).wait();

    printf("%f\n", sum[0]);
    return 0;
}
