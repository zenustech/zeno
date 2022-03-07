#include <zeno/back/allocator.h>
#include <zeno/utils/disable_copy.h>
#include <vector>
#include <cstdio>
#include <map>

namespace zeno {

struct fvector {
    std::vector<float, zallocator> m_data;
};

struct fvector_view {
    fvector &m_arr;

    fvector_view(fvector &arr) : m_arr(arr) {
    }

    float &operator[](std::size_t idx) const {
        return m_arr[idx];
    }

    float &at(std::size_t idx) const {
        return m_arr.at(idx);
    }

    auto begin() const {
        return m_arr.begin();
    }

    auto end() const {
        return m_arr.end();
    }

    float &front() const {
        return m_arr.front();
    }

    float &back() const {
        return m_arr.back();
    }

    std::size_t size() const {
        return m_arr.size();
    }

    float *data() const {
        return m_arr.data();
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

    auto grow_by(std::size_t n) const {
        auto ret = m_arr.end();
        m_arr.resize(m_arr.size() + n);
        return ret;
    }

    void pop_back() const {
        m_arr.pop_back();
    }

    ~fvector_view() {
    }
};

struct fvector_const_view {
    fvector const &m_arr;

    fvector_const_view(fvector const &arr) : m_arr(arr) {
    }

    float const &operator[](std::size_t idx) const {
        return m_arr[idx];
    }

    float const &at(std::size_t idx) const {
        return m_arr.at(idx);
    }

    auto begin() const {
        return m_arr.begin();
    }

    auto end() const {
        return m_arr.end();
    }

    float const &front() const {
        return m_arr.front();
    }

    float const &back() const {
        return m_arr.back();
    }

    std::size_t size() const {
        return m_arr.size();
    }

    float const *data() const {
        return m_arr.data();
    }
};

struct primitive {
    struct AttrInfo {
        fvector m_arr;
        std::string m_name;
    };

    std::vector<AttrInfo> m_attrs;

    constexpr fvector_view attr(std::size_t idx) {
        return m_attrs[idx].m_arr;
    }

    constexpr fvector_const_view attr(std::size_t idx) const {
        return m_attrs[idx].m_arr;
    }

    std::size_t lookup(std::string const &name) const {
        for (std::size_t idx = 0; idx < m_attrs.size(); idx++) {
            if (m_attrs[idx].m_name == name)
                return idx;
        }
        throw;
    }

    constexpr fvector_view attr(std::string const &name) {
        return attr(lookup(name));
    }

    constexpr fvector_const_view attr(std::string const &name) const {
        return attr(lookup(name));
    }
};

}

int main() {
    std::vector<int, zeno::zallocator> arr;
    arr.resize(1024 * 1024 * 1024);
    printf("%p\n", arr.data());
    return 0;
}
