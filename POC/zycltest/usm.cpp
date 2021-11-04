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

    constexpr decltype(auto) operator[](size_type index) const noexcept {
        return _M_data[index];
    }

    constexpr pointer_type data() const noexcept {
        return _M_data;
    }

    constexpr size_type size() const noexcept {
        return _M_size;
    }
};

span(has_span auto &&t) -> span<decltype(t.data()), decltype(t.size())>;


template <class T>
using vector = std::vector<T, allocator<T>>;


template <usize N>
struct range {
    sycl::range<N> _M_base;

    range() = default;

    range(usize i) requires (N == 1)
        : _M_base({i})
    {
    }

    constexpr range(sycl::range<N> const &base)
        : _M_base(base)
    {
    }

    constexpr operator auto const &() const {
        return _M_base;
    }

    constexpr usize const &operator[](usize i) const {
        return _M_base[i];
    }

    constexpr usize &operator[](usize i) {
        return _M_base[i];
    }
};


template <usize N>
struct nd_range {
    sycl::nd_range<N> _M_base;

    nd_range() = default;

    nd_range(range<N> const &global, range<N> const &local)
        : nd_range(sycl::nd_range<N>((sycl::range<N>)global,
                                     (sycl::range<N>)local))
    {
    }

    constexpr nd_range(sycl::nd_range<N> const &base)
        : _M_base(base)
    {
    }

    constexpr operator auto const &() const {
        return _M_base;
    }

    constexpr usize const &operator[](usize i) const {
        return _M_base[i];
    }

    constexpr usize &operator[](usize i) {
        return _M_base[i];
    }
};


template <usize N>
struct item {
    sycl::item<N> const &_M_that;

    item() = default;

    constexpr item(sycl::item<N> const &that)
        : _M_that(that)
    {
    }

    constexpr operator auto const &() const {
        return _M_that;
    }

    constexpr usize operator[](usize i) const {
        return this->get_id(i);
    }

    constexpr range<N> get_range() const {
        return range<N>(_M_that.get_range());
    }

    constexpr range<N> get_id() const {
        return range<N>(_M_that.get_range());
    }

    constexpr usize get_range(usize i) const {
        return _M_that.get_range(i);
    }

    constexpr usize get_id(usize i) const {
        return _M_that.get_id(i);
    }
};


template <usize N>
struct nd_item {
    sycl::nd_item<N> const &_M_that;

    nd_item() = default;

    constexpr nd_item(sycl::nd_item<N> const &that)
        : _M_that(that)
    {
    }

    constexpr operator auto const &() const {
        return _M_that;
    }

    constexpr usize operator[](usize i) const {
        return this->get_global_id(i);
    }

    constexpr range<N> get_global_range() const {
        return range<N>(_M_that.get_global_range());
    }

    constexpr range<N> get_block_range() const {
        return range<N>(_M_that.get_group_range());
    }

    constexpr range<N> get_local_range() const {
        return range<N>(_M_that.get_local_range());
    }

    constexpr range<N> get_global_id() const {
        return range<N>(_M_that.get_global_id());
    }

    constexpr range<N> get_block_id() const {
        return range<N>(_M_that.get_group());
    }

    constexpr range<N> get_local_id() const {
        return range<N>(_M_that.get_local_id());
    }

    constexpr usize get_global_range(usize i) const {
        return _M_that.get_global_range(i);
    }

    constexpr usize get_block_range(usize i) const {
        return _M_that.get_group_range(i);
    }

    constexpr usize get_local_range(usize i) const {
        return _M_that.get_local_range(i);
    }

    constexpr usize get_global_id(usize i) const {
        return _M_that.get_global_id(i);
    }

    constexpr usize get_block_id(usize i) const {
        return _M_that.get_group(i);
    }

    constexpr usize get_local_id(usize i) const {
        return _M_that.get_local_id(i);
    }
};


template <usize N>
void parallel_for(range<N> shape, auto &&body) {
    default_queue().submit([&] (sycl::handler &cgh) {
        cgh.parallel_for
            ( (sycl::range<N>)shape
            , [=] (sycl::item<N> it_) {
                item<N> const it(it_);
                body(it);
            });
    });
}

inline void synchronize() {
    default_queue().wait();
}


template <class T, usize N>
void parallel_reduce(nd_range<N> shape, T *out, T ident, auto &&binop, auto &&body) {
    auto e = default_queue().submit([&] (sycl::handler &cgh) {
        cgh.parallel_for
            ( (sycl::nd_range<N>)shape
            , sycl::reduction(out, ident, binop, sycl::property::reduction::initialize_to_identity{})
            , [=] (sycl::nd_item<N> it_, auto &reducer) {
                nd_item<N> const it(it_);
                body(it, reducer);
            });
    });
}


}


int main() {
    zpc::vector<float> arr(128);
    for (auto &a: arr) {
        a = drand48();
    }
    zpc::span varr = arr;

    zpc::parallel_for(zpc::range<1>(arr.size()), [=] (zpc::item<1> it) {
        varr[it[0]] = it[0];
    });

    zpc::vector<float> out(1);
    zpc::parallel_reduce(zpc::nd_range<1>(arr.size(), 8), out.data(), 0.f, std::plus{}, [=] (zpc::nd_item<1> it, auto &reducer) {
        reducer += varr[it[0]];
    });

    zpc::synchronize();
    std::cout << out[0] << std::endl;

    return 0;
}
