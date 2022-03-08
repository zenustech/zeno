#include <zeno/back/allocator.h>
#include <zeno/utils/vec.h>
#include <cstdio>
#include <vector>
#include <string>
#include <array>

namespace zeno {

using fvector = std::vector<float, allocator<float>>;

struct primitive;

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

    float read(std::size_t idx) const {
        return m_arr[idx];
    }

    void write(std::size_t idx, float val) const {
        m_arr[idx] = val;
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

struct fvector_const_view {
    fvector const &m_arr;

    using value_type = float;

    fvector_const_view(std::piecewise_construct_t, fvector const &arr)
        : m_arr(arr) {
    }

    float const &operator[](std::size_t idx) const {
        return m_arr[idx];
    }

    float const &at(std::size_t idx) const {
        return m_arr.at(idx);
    }

    float read(std::size_t idx) const {
        return m_arr[idx];
    }

    std::size_t size() const {
        return m_arr.size();
    }

    float const *data() const {
        return m_arr.data();
    }

    auto begin() const {
        return m_arr.begin();
    }

    auto end() const {
        return m_arr.end();
    }

    fvector const &original_vector() const {
        return m_arr;
    }
};

template <std::size_t N>
struct fvector_multiview {
    static_assert(N > 0);
    std::array<fvector_view, N> m_views;

    using value_type = vec<N, float>;

    std::size_t size() const {
        return m_views[0].size();
    }

    vec<N, float> read(std::size_t idx) const {
        vec<N, float> val;
        for (std::size_t i = 0; i < N; i++) {
            val[i] = m_views[i][idx];
        }
        return val;
    }

    void write(std::size_t idx, vec<N, float> const &val) const {
        for (std::size_t i = 0; i < N; i++) {
            m_views[i][idx] = val[i];
        }
    }

    struct _soa_reference {
        fvector_multiview &m_that;
        std::size_t idx;

        vec<N, float> get() const {
            return m_that.read(idx);
        }

        operator vec<N, float>() const {
            return get();
        }

        _soa_reference &operator=(vec<N, float> const &val) const {
            m_that.write(idx, val);
            return *this;
        }
    };

    _soa_reference operator[](std::size_t idx) const {
        return {*this, idx};
    }

    _soa_reference at(std::size_t idx) const {
        (void)m_views[0].at(idx);
        return {*this, idx};
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

struct primitive {
    struct AttrInfo {
        fvector m_arr;
        std::string m_name;
    };

    std::vector<AttrInfo> m_attrs;

    primitive() : m_attrs(1) {
        m_attrs[0].m_name = "_";
    }

    fvector_view attr(std::size_t idx) {
        return {std::piecewise_construct, m_attrs.at(idx).m_arr};
    }

    fvector_const_view attr(std::size_t idx) const {
        return {std::piecewise_construct, m_attrs.at(idx).m_arr};
    }

    std::size_t lookup(std::string const &name) const {
        for (std::size_t idx = 0; idx < m_attrs.size(); idx++) {
            if (m_attrs[idx].m_name == name)
                return idx;
        }
        throw;
    }

    fvector_view attr(std::string const &name) {
        return attr(lookup(name));
    }

    fvector_const_view attr(std::string const &name) const {
        return attr(lookup(name));
    }

    std::size_t num_attrs() const {
        return m_attrs.size();
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
    primitive prim;
    auto pos = prim.attr("_");
    return 0;
}
