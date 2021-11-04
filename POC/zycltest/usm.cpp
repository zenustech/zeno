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
    std::array<usize, N> _M_base{};

    range() = default;

    constexpr range(usize i) requires (N == 1)
        : _M_base({i})
    {
    }

    constexpr range(std::array<usize, N> const &arr)
        : _M_base(arr)
    {
    }

    constexpr range(sycl::range<N> const &base)
        : _M_base([] <std::size_t ...Is>
        (auto &x, std::index_sequence<Is...>) {
            return std::array<usize, N>(x[Is]...);
        }(_M_base, std::make_index_sequence<N>{}))
    {
    }

    constexpr operator sycl::range<N>() const {
        return [] <std::size_t ...Is>
        (auto &x, std::index_sequence<Is...>) {
            return sycl::range<N>(std::get<Is>(x)...);
        }(_M_base, std::make_index_sequence<N>{});
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
    range<N> _M_global;
    range<N> _M_local;

    nd_range() = default;

    constexpr nd_range(range<N> const &global, range<N> const &local)
        : _M_global(global)
        , _M_local(local)
    {
    }

    constexpr nd_range(sycl::nd_range<N> const &base)
    {
    }

    constexpr operator sycl::nd_range<N>() const {
        return { (sycl::range<N>)_M_global
               , (sycl::range<N>)_M_local
               };
    }

    constexpr usize const &operator[](usize i) const {
        return _M_global[i];
    }

    constexpr usize &operator[](usize i) {
        return _M_global[i];
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


inline void synchronize() {
    default_queue().wait();
}


template <usize N>
void parallel_nd_for
    ( nd_range<N> shape
    , auto &&body
    , auto &&...args
    ) {
    default_queue().submit
    ( [&] (sycl::handler &cgh) {
        cgh.parallel_for
            ( (sycl::nd_range<N>)shape
            , std::forward<decltype(args)>(args)...
            , [=] (sycl::nd_item<N> it_, auto &...args) {
                nd_item<N> const it(it_);
                body(it, args...);
            });
    });
}


template <usize N>
void parallel_for
    ( range<N> shape
    , range<N> local_dim
    , auto &&body
    , auto &&...args
    ) {
    range<N> global_dim;
    for (usize i = 0; i < N; i++) {
        global_dim[i] = std::max((usize)1, (shape[i] + local_dim[i] - 1) / local_dim[i] * local_dim[i]);
        local_dim[i] = std::clamp(local_dim[i], (usize)1, shape[i]);
    }
    nd_range<N> nd_shape
        ( global_dim
        , local_dim
        );
    parallel_nd_for
    ( nd_shape
    , [=]
    ( nd_item<N> const &it
    , auto &...args) {
        for (usize i = 0; i < N; i++) {
            [[unlikely]] if (it[i] > shape[i])
                return;
        }
        body(it, args...);
    }
    , std::forward<decltype(args)>(args)...
    );
}


template <class T>
auto reduction
        ( T *out
        , T ident
        , auto &&binop
        ) {
    sycl::property::reduction::initialize_to_identity props;
    return sycl::reduction(out, ident, binop, props);
}


}


int main() {
    zpc::vector<float> arr(100);
    for (auto &a: arr) {
        a = drand48();
    }
    zpc::span varr = arr;

    zpc::parallel_for
    ( zpc::range<1>(arr.size())
    , zpc::range<1>(8)
    , [=] (zpc::nd_item<1> it) {
        varr[it[0]] = it[0];
    });

    zpc::vector<float> out(1);
    zpc::parallel_for
    ( zpc::range<1>(arr.size())
    , zpc::range<1>(8)
    , [=] (zpc::nd_item<1> it, auto &reducer) {
        reducer.combine(varr[it[0]]);
    }
    , zpc::reduction(out.data(), 0.f, [] (auto x, auto y) { return x + y; })
    );

    zpc::synchronize();
    std::cout << out[0] << std::endl;

    return 0;
}
