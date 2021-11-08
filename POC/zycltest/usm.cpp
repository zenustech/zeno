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
struct vector : std::vector<T, allocator<T>> {
    using std::vector<T, allocator<T>>::vector;
};


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
    sycl::nd_item<N> const &_M_that;

    item() = default;

    constexpr item(sycl::nd_item<N> const &that)
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


namespace details {

template <usize N>
sycl::nd_range<N> calc_nd_range(range<N> const &dim, range<N> local_dim) {
    range<N> global_dim;
    for (usize i = 0; i < N; i++) {
        global_dim[i] = std::max((usize)1, (dim[i] + local_dim[i] - 1) / local_dim[i] * local_dim[i]);
        local_dim[i] = std::max((usize)1, std::min(local_dim[i], dim[i]));
    }
    return
        { (sycl::range<N>)global_dim
        , (sycl::range<N>)local_dim
        };
}

template <class T>
concept is_our_element = requires (T t) {
    t(std::declval<sycl::handler &>());
};

template <bool is_ours>
inline constexpr std::tuple<> select_our_elements() {
    return {};
}

template <bool is_ours>
inline constexpr auto select_our_elements(auto &&head, auto &&...args) {
    if constexpr (is_our_element<std::remove_cvref_t<decltype(head)>> == is_ours) {
        return std::apply([&] (auto &...args) {
            return std::forward_as_tuple(head, args...);
        }, select_our_elements<is_ours>(args...));
    } else {
        return select_our_elements<is_ours>(args...);
    }
}

template <std::size_t I, class ArgTypes>
inline constexpr auto count_trues_before() {
    return [&] <std::size_t ...Js> (std::index_sequence<Js...>) {
        return ((std::size_t)std::tuple_element_t<Js, ArgTypes>::value + ... + 0);
    }(std::make_index_sequence<I>{});
}

template <std::size_t I, class ArgTypes>
inline constexpr decltype(auto) get_shuffled_at(auto &&it, auto &&our_args, auto &&sycl_args) {
    constexpr auto O = count_trues_before<I, ArgTypes>();
    if constexpr (std::tuple_element_t<I, ArgTypes>::value) {
        return std::get<O>(our_args)(it);
    } else {
        return std::get<I - O>(sycl_args);
    }
}

template <class ArgTypes>
inline constexpr auto shuffle_element_indices(auto &&it, auto &&our_args, auto &&sycl_args) {
    return [&] <std::size_t ...Is> (std::index_sequence<Is...>) {
        return std::forward_as_tuple(get_shuffled_at<Is, ArgTypes>(it, our_args, sycl_args)...);
    }(std::make_index_sequence<std::tuple_size_v<ArgTypes>>{});
}

template <usize N>
inline bool is_inside_range(sycl::nd_item<N> const &it, range<N> const &dim) {
    return [&] <std::size_t ...Is> (std::index_sequence<Is...>) {
        return ((it.get_global_id(Is) < dim[Is]) && ...);
    }(std::make_index_sequence<N>{});
}

}


template
< class Kern = void
, bool IsClamped = true
, usize N>
auto parallel_for
    ( range<N> dim
    , range<N> local_dim
    , auto &&body
    , auto &&...args
    ) {
    using Key = std::conditional_t<
        std::is_void_v<Kern>,
        std::remove_cvref_t<decltype(body)>,
        Kern>;
    auto nd_dim = details::calc_nd_range<N>(dim, local_dim);
    using ArgTypes = std::tuple<std::bool_constant<
          details::is_our_element<std::remove_cvref_t<decltype(args)>>
          >...>;

    auto our_args = details::select_our_elements<true>(args...);
    return std::apply([&] (auto &...sycl_args) {
        return default_queue().submit([&] (sycl::handler &cgh) {

            auto our_data = std::apply([&] (auto &...our_args) {
                return std::tuple(our_args(cgh)...);
            }, our_args);

            cgh.parallel_for<Key>
            ( nd_dim
            , sycl_args...
            , [=]
            ( sycl::nd_item<N> const &it_
            , auto &...sycl_args) {
                if constexpr (IsClamped) {
                    [[unlikely]] if (!details::is_inside_range<N>(it_, dim))
                        return;
                }
                item<N> const it(it_);

                std::apply([&] (auto &&...args) {
                    body(it, std::forward<decltype(args)>(args)...);
                }, details::shuffle_element_indices<ArgTypes>(
                    it, our_data, std::forward_as_tuple(sycl_args...)));
            });

        });
    }, details::select_our_elements<false>(args...));
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


template <class T, usize N>
auto local_buffer(range<N> shape) {
    return [=] (sycl::handler &cgh) {
        sycl::accessor
        < T
        , N
        , sycl::access::mode::read_write
        , sycl::access::target::local
        > acc
        ( (sycl::range<N>)shape
        , cgh
        );
        return [acc = std::move(acc)] (auto &it) {
            return acc.get_pointer();
        };
    };
}


template <class T>
auto local_buffer() {
    return [=] (sycl::handler &cgh) {
        sycl::accessor
        < T
        , 1
        , sycl::access::mode::read_write
        , sycl::access::target::local
        > acc
        ( sycl::range<1>(1)
        , cgh
        );
        return [acc = std::move(acc)] (auto &it) -> auto & {
            return *acc.get_pointer();
        };
    };
}


}


int main() {
    zpc::vector<float> arr(128);

    zpc::span varr = arr;

    zpc::parallel_for
    ( zpc::range<1>(arr.size())
    , zpc::range<1>(8)
    , [=] (zpc::item<1> it) {
        varr[it[0]] = it[0] + 1;
    }).wait();

    zpc::vector<float> out(1);
    zpc::parallel_for
    ( zpc::range<1>(arr.size())
    , zpc::range<1>(8)
    , [=] (zpc::item<1> it, auto bufptr, auto &reducer) {
        reducer.combine(varr[it[0]]);
    }
    , zpc::local_buffer<float>(zpc::range<1>(1))
    , zpc::reduction(out.data(), 0.f, [] (auto x, auto y) { return x + y; })
    ).wait();

    std::cout << out[0] << std::endl;

    return 0;
}
