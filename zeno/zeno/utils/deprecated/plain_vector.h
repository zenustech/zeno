#pragma once

#include <vector>
#include <type_traits>
#include <zeno/zeno/utils/vec.h>
#include <zeno/zeno/utils/allocator.h>

namespace zeno {

template <class T>
using plain_vector = std::vector<T, allocator<T, 64, true>>;

template <class T>
using plain_vector_ref = plain_vector<T> &;

template <class T, std::size_t N>
struct attr_vector_view {
    static_assert(std::is_same_v<std::decay_t<T>, T>);

    std::array<plain_vector_ref<T>, N> m_arr;

    using component_type = T;
    using value_type = vec<N, component_type>;

    constexpr void push_back(value_type const &t) {
        for (int i = 0; i < N; i++)
            m_arr[i].push_back(t[i]);
    }

    constexpr void push_back(value_type &&t) {
        for (int i = 0; i < N; i++)
            m_arr[i].push_back(std::move(t[i]));
    }

    template <class ...Ts>
    constexpr void emplace_back(Ts &&...ts) {
        this->push_back(value_type{std::forward<Ts>(ts)});
    }

    constexpr size_t size() const {
        return m_arr[0].size();
    }

    constexpr size_t capacity() const {
        return m_arr[0].capacity();
    }

    constexpr size_t resize(size_t siz) {
        for (int i = 0; i < N; i++)
            m_arr[i].resize(siz);
    }

    constexpr size_t reserve(size_t siz) {
        for (int i = 0; i < N; i++)
            m_arr[i].reserve(siz);
    }

    constexpr size_t shrink_to_fit() {
        for (int i = 0; i < N; i++)
            m_arr[i].shrink_to_fit(siz);
    }
};

template <class T>
struct attr_vector {
    static_assert(std::is_same_v<std::decay_t<T>, T>);

    struct VecData {
        plain_vector<T> m_arr;
        std::string m_name;
    };

    std::vector<VecData> m_chs;

    template <std::size_t N, std::size_t ...is>
    constexpr static std::array<plain_vector_ref<T>, N> _view_helper(
        std::array<std::size_t, N> const &inds, std::index_sequence<Is...>) {
        return {m_chs.at(inds[Is]).m_arr...};
    }

    template <std::size_t N>
    constexpr attr_vector_view<T, N> view(std::array<std::size_t, N> const &inds) {
        return {_view_helper(inds, std::make_index_sequence<N>{})};
    }

    template <std::size_t N>
    constexpr attr_vector_view<T, N> view(std::array<std::string, N> const &inds) {
        return view(lookup(inds));
    }

    constexpr std::size_t lookup(std::string const &name) const {
        for (std::size_t i = 0; i < m_chs.size(); i++) {
            if (m_chs[i].name == name)
                return i;
        }
        throw;
    }

    template <std::size_t n, std::size_t ...is>
    constexpr std::array<std::size_t, N> &_lookup_helper(
        std::array<std::string, N> const &names, std::index_sequence<Is...>) const {
        return {lookup(names[Is])...};
    }

    constexpr std::array<std::size_t, N> &lookup(std::array<std::string, N> const &names) const {
        _lookup_helper(names, std::make_index_sequence<N>{});
    }
};

}
