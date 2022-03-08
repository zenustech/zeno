#include <zeno/back/allocator.h>
#include <zeno/utils/vec.h>
#include <cstdio>
#include <vector>
#include <string>
#include <array>

namespace zeno {

using fvector = std::vector<float, allocator<float>>;

struct fvector_view {
    fvector &m_arr;

    using value_type = float;

    fvector_view(std::piecewise_construct_t, fvector &arr)
        : m_arr(arr) {
    }

    float &operator[](std::size_t idx) const {
        return m_arr[idx];
    }

    float &at(std::size_t idx) const {
        return m_arr.at(idx);
    }

    std::size_t size() const {
        return m_arr.size();
    }

    float *data() const {
        return m_arr.data();
    }

    auto begin() const {
        return m_arr.begin();
    }

    auto end() const {
        return m_arr.end();
    }

    void clear() const {
        m_arr.clear();
    }

    void resize(std::size_t n) const {
        m_arr.resize(n);
    }

    std::size_t capacity() const {
        return m_arr.capacity();
    }

    void reserve(std::size_t n) const {
        m_arr.reserve(n);
    }

    void shrink_to_fit() const {
        m_arr.shrink_to_fit();
    }

    void push_back(float val) const {
        m_arr.push_back(val);
    }

    template <class ...Ts>
    void emplace_back(Ts &&...ts) const {
        this->push_back(value_type(std::forward<Ts>(ts)...));
    }

    void pop_back() const {
        m_arr.pop_back();
    }

    fvector &original_vector() const {
        return m_arr;
    }
};

template <std::size_t N>
struct fvector_soa_view {
    static_assert(N > 0);
    std::array<fvector_view, N> m_views;

    fvector_soa_view(std::piecewise_construct_t, std::array<fvector_view, N> &&views)
        : m_views(std::move(views)) {
    }

    using value_type = vec<N, float>;

    std::size_t size() const {
        return m_views[0].size();
    }

    struct vec_reference {
        std::array<float *, N> m_ptrs;

        explicit vec_reference(std::piecewise_construct_t) {}

        float &operator[](std::size_t i) const {
            return *m_ptrs[i];
        }

        operator vec<N, float>() const {
            vec<N, float> val;
            for (std::size_t i = 0; i < N; i++) {
                val[i] = *m_ptrs[i];
            }
            return val;
        }

        vec_reference &operator=(vec<N, float> const &val) const {
            for (std::size_t i = 0; i < N; i++) {
                *m_ptrs[i] = val[i];
            }
            return *this;
        }
    };

    vec_reference operator[](std::size_t idx) const {
        vec_reference vref{std::piecewise_construct};
        for (std::size_t i = 0; i < N; i++) {
            vref.m_ptrs[i] = &m_views[i][idx];
        }
        return vref;
    }

    vec_reference at(std::size_t idx) const {
        (void)m_views[0].at(idx);
        return this->operator[](idx);
    }

    void push_back(vec<N, float> const &val) const {
        for (std::size_t i = 0; i < N; i++) {
            m_views[i].push_back(val[i]);
        }
    }

    template <class ...Ts>
    void emplace_back(Ts &&...ts) const {
        this->push_back(value_type(std::forward<Ts>(ts)...));
    }

    void pop_back() const {
        for (std::size_t i = 0; i < N; i++) {
            m_views[i].pop_back();
        }
    }

    void clear() const {
        for (std::size_t i = 0; i < N; i++) {
            m_views[i].clear();
        }
    }

    void resize(std::size_t n) const {
        for (std::size_t i = 0; i < N; i++) {
            m_views[i].resize(n);
        }
    }

    std::size_t capacity() const {
        return m_views[0].capacity();
    }

    void reserve(std::size_t n) const {
        for (std::size_t i = 0; i < N; i++) {
            m_views[i].reserve(n);
        }
    }

    void shrink_to_fit() const {
        for (std::size_t i = 0; i < N; i++) {
            m_views[i].shrink_to_fit();
        }
    }

    fvector_view const &component(std::size_t idx) const {
        return m_views.at(idx);
    }

    template <std::size_t I>
    fvector_view const &component() const {
        return std::get<I>(m_views);
    }
};

struct attrvector {
    struct AttrInfo {
        mutable fvector m_arr;
        std::string m_name;
    };

    std::vector<AttrInfo> m_attrs;

    fvector_view attr(std::size_t idx) const {
        return {std::piecewise_construct, m_attrs.at(idx).m_arr};
    }

    std::size_t lookup(std::string const &name) const {
        for (std::size_t idx = 0; idx < m_attrs.size(); idx++) {
            if (m_attrs[idx].m_name == name)
                return idx;
        }
        throw;
    }

    fvector_view attr(std::string const &name) const {
        return attr(lookup(name));
    }

    template <std::size_t N, std::size_t ...Is>
    fvector_soa_view<N> _helper_attr(std::string const &name, std::index_sequence<Is...>) const {
        return {std::piecewise_construct, {attr(name + '.' + std::to_string(Is))...}};
    }

    template <std::size_t N>
    fvector_soa_view<N> attr(std::string const &name) const {
        return _helper_attr<N>(name, std::make_index_sequence<N>{});
    }

    std::size_t num_attrs() const {
        return m_attrs.size();
    }

    fvector_view add_attr(std::string const &name) {
        auto back = m_attrs.emplace_back();
        back.m_name = name;
        return {std::piecewise_construct, back.m_arr};
    }

    template <std::size_t N, std::size_t ...Is>
    fvector_soa_view<N> _helper_add_attr(std::string const &name, std::index_sequence<Is...>) {
        return {std::piecewise_construct, {add_attr(name + '.' + std::to_string(Is))...}};
    }

    template <std::size_t N>
    fvector_soa_view<N> add_attr(std::string const &name) {
        return _helper_add_attr<N>(name, std::make_index_sequence<N>{});
    }

    std::size_t size() const {
        return m_attrs.at(0).m_arr.size();
    }

    void resize(std::size_t n) {
        for (std::size_t i = 0; i < m_attrs.size(); i++) {
            m_attrs[i].m_arr.resize(n);
        }
    }

    void reserve(std::size_t n) {
        for (std::size_t i = 0; i < m_attrs.size(); i++) {
            m_attrs[i].m_arr.reserve(n);
        }
    }

    void _check_sync_size() {
        std::size_t n = m_attrs.at(0).m_arr.size();
        for (std::size_t i = 1; i < m_attrs.size(); i++) {
            n = std::max(n, m_attrs[i].m_arr.size());
        }
        resize(n);
    }
};

}

using namespace zeno;

int main() {
    attrvector prim;
    auto pos = prim.add_attr("pos");
    pos.resize(32);
    printf("%f\n", pos[0]);
    return 0;
}
