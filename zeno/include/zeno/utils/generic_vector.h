#pragma once

#include "fast_allocator.h"
#include "bit_operations.h"
#include "variantswitch.h"
#include <type_traits>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>
#include <array>
#include <tuple>
#include <new>

namespace zeno {
inline namespace generic_vector {

namespace details {

struct detailed_constructor { explicit detailed_constructor() = default; };

struct byte_vector {
    std::byte *m_data;
    std::size_t m_size;
    std::uint8_t m_shift;
    fast_allocator<std::byte> m_alloc;

    struct initialized_tag {
        explicit initialized_tag() = default;
    };

    byte_vector()
        : m_data{nullptr}
        , m_shift{0}
        , m_size{0}
    {
    }

    explicit byte_vector(std::size_t n) {
        m_shift = ceil_log2(n);
        m_data = m_alloc.allocate(1 << m_shift);
        m_size = n;
    }

    explicit byte_vector(std::size_t n, initialized_tag) {
        m_shift = ceil_log2(n);
        m_data = m_alloc.allocate(1 << m_shift);
        std::memset(m_data, 0, n);
        m_size = n;
    }

    byte_vector(byte_vector const &that)
        : byte_vector{that.m_size}
    {
        if (m_size)
            std::memcpy(m_data, that.m_data, m_size);
    }

    ~byte_vector() {
        if (m_shift)
            m_alloc.deallocate(m_data, m_shift);
    }

    byte_vector &operator=(byte_vector const &that)
    {
        if (this == &that)
            return *this;
        resize(that.m_size);
        if (m_size)
            std::memcpy(m_data, that.m_data, m_size);
        return *this;
    }

    byte_vector(byte_vector &&that)
        : m_data{that.m_data}
        , m_size{that.m_size}
        , m_shift{that.m_shift}
        , m_alloc{std::move(that.m_alloc)}
    {
        that.m_data = nullptr;
        that.m_size = 0;
        that.m_shift = 0;
    }

    byte_vector &operator=(byte_vector &&that)
    {
        if (this == &that)
            return *this;
        m_data = that.m_data;
        m_size = that.m_size;
        m_shift = that.m_shift;
        m_alloc = std::move(that.m_alloc);
        that.m_data = nullptr;
        that.m_size = 0;
        that.m_shift = 0;
        return *this;
    }

    std::byte *data() {
        return m_data;
    }

    std::byte const *data() const {
        return m_data;
    }

    std::size_t capacity() const {
        return 1 << m_shift;
    }

    void reserve(std::size_t n) {
        if (capacity() >= n)
            return;
        std::uint8_t shift = ceil_log2(n);
        if (m_shift >= shift)
            return;
        std::byte *data = m_alloc.allocate(1 << shift);
        if (m_size)
            std::memcpy(data, m_data, m_size);
        if (m_shift)
            m_alloc.deallocate(m_data, 1 << m_shift);
        m_data = data;
        m_shift = shift;
    }

    void shrink_to_fit() {
        std::uint8_t shift = ceil_log2(m_size);
        if (m_shift <= shift)
            return;
        std::byte *data = m_alloc.allocate(1 << shift);
        if (m_size)
            std::memcpy(data, m_data, m_size);
        if (m_shift)
            m_alloc.deallocate(m_data, 1 << m_shift);
        m_data = data;
        m_shift = shift;
    }

    std::size_t size() const {
        return m_size;
    }

    void clear() {
        m_size = 0;
    }

    void resize(std::size_t n) {
        reserve(n);
        m_size = n;
    }

    void resize(std::size_t n, initialized_tag) {
        reserve(n);
        if (n > m_size)
            std::memset(m_data + m_size, 0, n - m_size);
        m_size = n;
    }
};

template <class T>
struct vector_view {
    byte_vector *m_origin;
    using value_type = T;

    explicit vector_view(byte_vector *origin)
        : m_origin{origin} {}

    T *data() const {
        return (T *)m_origin->data();
    }

    std::size_t size() const {
        return m_origin->size() / sizeof(T);
    }

    void resize(std::size_t n) const {
        return m_origin->resize(n * sizeof(T));
    }

    std::size_t capacity() const {
        return m_origin->capacity() / sizeof(T);
    }

    void reserve(std::size_t n) const {
        return m_origin->reserve(n * sizeof(T));
    }

    void shrink_to_fit() const {
        return m_origin->shrink_to_fit();
    }

    T *grow_by(std::size_t n) const {
        std::size_t offset = size();
        m_origin->resize((offset + n) * sizeof(T));
        return (T *)m_origin->data() + offset;
    }

    T *push_back(T &&val) const {
        T *ptr = grow_by(1);
        ::new ((void *)ptr) T(std::move(val));
        return ptr;
    }

    T *push_back(T const &val) const {
        T *ptr = grow_by(1);
        ::new ((void *)ptr) T(val);
        return ptr;
    }
};

/*template <class ...Ts>
struct multi_vector_view {
    using value_type = std::tuple<Ts...>;
    inline static constexpr std::size_t kNumComps = sizeof...(Ts);

    std::tuple<vector_view<Ts>...> m_origin;

    multi_vector_view(vector_view<Ts> const &...views)
        : m_origin{views...} {
    }

    template <std::size_t I>
    std::tuple_element_t<I, std::tuple<Ts...>> *data() const {
        return std::get<I>(m_origin).data();
    }

    template <std::size_t ...Is>
    std::size_t _helper_size(std::index_sequence<Is...>) const {
        return std::min({std::get<Is>(m_origin).size()...});
    }

    std::size_t size() const {
        return _helper_size(std::make_index_sequence<kNumComps>{});
    }

    template <std::size_t ...Is>
    void _helper_resize(std::size_t n, std::index_sequence<Is...>) const {
        ((std::get<Is>(m_origin).resize(n), 0), ...);
    }

    void resize(std::size_t n) const {
        return _helper_resize(n, std::make_index_sequence<kNumComps>{});
    }

    template <std::size_t ...Is>
    std::size_t _helper_capacity(std::index_sequence<Is...>) const {
        return std::min({std::get<Is>(m_origin).capacity()...});
    }

    std::size_t capacity() const {
        return _helper_capacity(std::make_index_sequence<N>{});
    }

    template <std::size_t ...Is>
    void _helper_reserve(std::size_t n, std::index_sequence<Is...>) const {
        ((std::get<Is>(m_origin).reserve(n), 0), ...);
    }

    void reserve(std::size_t n) const {
        return _helper_reserve(n, std::make_index_sequence<kNumComps>{});
    }

    template <std::size_t ...Is>
    void _helper_shrink_to_fit(std::index_sequence<Is...>) const {
        ((std::get<Is>(m_origin).shrink_to_fit(), 0), ...);
    }

    void shrink_to_fit() const {
        return _helper_reserve(std::make_index_sequence<kNumComps>{});
    }

    template <std::size_t ...Is>
    std::tuple<Ts *...> _helper_grow_by(std::size_t n, std::index_sequence<Is...>) const {
        return {std::get<Is>(m_origin).grow_by(n)...};
    }

    std::tuple<Ts *...> grow_by(std::size_t n) const {
        return _helper_grow_by(n, std::make_index_sequence<kNumComps>{});
    }

    template <std::size_t ...Is>
    std::tuple<Ts *...> _helper_push_back(std::tuple<Ts...> const &val, std::index_sequence<Is...>) const {
        return {std::get<Is>(m_origin).push_back(std::get<Is>(val))...};
    }

    std::tuple<Ts *...> push_back(std::tuple<Ts...> const &val) const {
        return _helper_push_back(val, std::make_index_sequence<kNumComps>{});
    }

    std::tuple<Ts *...> emplace_back(Ts const &...vals) const {
        return push_back({vals...});
    }
};

template <class ...Ts>
multi_vector_view(vector_view<Ts> const &...) -> multi_vector_view<Ts...>;*/

template <class T, std::size_t N>
struct comp_vector_view {
    using value_type = std::array<T, N>;

    std::array<vector_view<T>, N> m_origin;

    comp_vector_view(std::array<vector_view<T>, N> const &views, detailed_constructor)
        : m_origin{views} {
    }

    template <std::size_t I>
    T *data() const {
        return std::get<I>(m_origin).data();
    }

    template <std::size_t ...Is>
    std::size_t _helper_size(std::index_sequence<Is...>) const {
        return std::min({std::get<Is>(m_origin).size()...});
    }

    std::size_t size() const {
        return _helper_size(std::make_index_sequence<N>{});
    }

    template <std::size_t ...Is>
    void _helper_resize(std::size_t n, std::index_sequence<Is...>) const {
        ((std::get<Is>(m_origin).resize(n), 0), ...);
    }

    void resize(std::size_t n) const {
        return _helper_resize(n, std::make_index_sequence<N>{});
    }

    template <std::size_t ...Is>
    std::size_t _helper_capacity(std::index_sequence<Is...>) const {
        return std::min({std::get<Is>(m_origin).capacity()...});
    }

    std::size_t capacity() const {
        return _helper_capacity(std::make_index_sequence<N>{});
    }

    template <std::size_t ...Is>
    void _helper_reserve(std::size_t n, std::index_sequence<Is...>) const {
        ((std::get<Is>(m_origin).reserve(n), 0), ...);
    }

    void reserve(std::size_t n) const {
        return _helper_reserve(n, std::make_index_sequence<N>{});
    }

    template <std::size_t ...Is>
    void _helper_shrink_to_fit(std::index_sequence<Is...>) const {
        ((std::get<Is>(m_origin).shrink_to_fit(), 0), ...);
    }

    void shrink_to_fit() const {
        return _helper_reserve(std::make_index_sequence<N>{});
    }

    template <std::size_t ...Is>
    std::array<T *, N> _helper_grow_by(std::size_t n, std::index_sequence<Is...>) const {
        return {std::get<Is>(m_origin).grow_by(n)...};
    }

    std::array<T *, N> grow_by(std::size_t n) const {
        return _helper_grow_by(n, std::make_index_sequence<N>{});
    }

    template <std::size_t ...Is>
    std::array<T *, N> _helper_push_back(std::array<T, N> const &val, std::index_sequence<Is...>) const {
        return {std::get<Is>(m_origin).push_back(std::get<Is>(val))...};
    }

    std::array<T *, N> push_back(std::array<T, N> const &val) const {
        return _helper_push_back(val, std::make_index_sequence<N>{});
    }

    template <class ...Ts>
    std::array<T *, N> emplace_back(Ts const &...vals) const {
        return push_back(value_type(vals...));
    }
};

enum class dtype_e : std::uint8_t {
    Float = 0,
    Int = 1,
};

using dtype_e_variant = std::variant<float, int>;

dtype_e_variant dtype_to_variant(dtype_e dtype) {
    return enum_variant<dtype_e_variant>(dtype);
}

struct vector_storage {
    mutable byte_vector m_data;

    dtype_e m_dtype;

    dtype_e dtype() const { return m_dtype; }

    explicit vector_storage(dtype_e dt)
        : m_dtype(dt)
    {
    }

    template <class T>
    bool dtype_is() const {
        return std::visit([&] (auto val) -> bool {
            using Val = decltype(val);
            return std::is_same_v<Val, T>;
        }, dtype_to_variant(dtype()));
    }

    template <class T>
    vector_view<T> view() const {
        if (!dtype_is<T>())
            throw std::bad_variant_access{};
        return vector_view<T>{&m_data};
    }

    template <class Func>
    void visit_view(Func const &func) const {
        std::visit([&] (auto val) {
            using T = decltype(val);
            func(view<T>());
        }, dtype_to_variant(dtype()));
    }
};

template <class T>
struct array_traits : std::false_type {
    using value_type = T;
    inline static constexpr std::size_t dimension = 1;
};

template <class T, std::size_t N>
struct array_traits<std::array<T, N>> : std::true_type {
    using value_type = T;
    inline static constexpr std::size_t dimension = N;
};

template <class T>
using comp_vector_viewer = std::conditional_t<array_traits<T>::value,
      comp_vector_view<typename array_traits<T>::value_type,
      array_traits<T>::dimension>, vector_view<T>>;

struct comp_vector_storage {
    mutable std::vector<byte_vector> m_data;

    dtype_e m_dtype;

    dtype_e dtype() const { return m_dtype; }
    std::size_t dim() const { return m_data.size(); }

    explicit comp_vector_storage(dtype_e dt, std::size_t n = 1)
        : m_dtype(dt)
        , m_data(n)
    {
    }

    template <class T>
    explicit comp_vector_storage(std::in_place_type_t<T>)
        : m_dtype(variant_enum<dtype_e, dtype_e_variant,
                  typename array_traits<T>::value_type>::value)
        , m_data(array_traits<T>::dimension)
    {
    }

    template <class T>
    bool _comp_dtype_is() const {
        return std::visit([&] (auto val) -> bool {
            using Val = decltype(val);
            return std::is_same_v<Val, T>;
        }, dtype_to_variant(dtype()));
    }

    template <class T>
    bool dtype_is() const {
        if constexpr (array_traits<T>::value) {
            return dim() == array_traits<T>::dimension
                && _comp_dtype_is<typename array_traits<T>::value_type>();
        } else {
            return dim() == 1 && _comp_dtype_is<T>();
        }
    }

    template <class T, std::size_t N, std::size_t ...Is>
    comp_vector_view<T, N> _helper_comp_view(std::index_sequence<Is...>) const {
        return {std::array<vector_view<T>, N>{vector_view<T>{&m_data[Is]}...}, detailed_constructor{}};
    }

    template <class T>
    comp_vector_viewer<T> view() const {
        if constexpr (array_traits<T>::value) {
            return _comp_view<typename array_traits<T>::value_type,
                              array_traits<T>::dimension>();
        } else {
            if (!_comp_dtype_is<T>())
                throw std::bad_variant_access{};
            if (dim() != 1)
                throw std::bad_variant_access{};
            return vector_view<T>{&m_data[0]};
        }
    }

    template <class T, std::size_t N>
    comp_vector_view<T, N> _comp_view() const {
        if (!_comp_dtype_is<T>())
            throw std::bad_variant_access{};
        if (dim() != N)
            throw std::bad_variant_access{};
        return _helper_comp_view<T, N>(std::make_index_sequence<N>{});
    }

    template <class Func>
    void visit_view(Func const &func) const {
        std::visit([&] (auto val) {
            using T = decltype(val);
            index_switch<4>(dim() - 1, [&] (auto idx) {
                constexpr std::size_t N = idx.value + 1;
                if constexpr (N == 1)
                    func(view<T, N>());
                else
                    func(view<T>());
            });
        }, dtype_to_variant(dtype()));
    }
};

}

using details::comp_vector_storage;
using details::comp_vector_viewer;
using details::dtype_e;

}
}
