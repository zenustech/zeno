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

    ~fvector_view() {
    }
};

struct fvector_const_view {
    fvector const &m_arr;

    fvector_const_view(fvector const &arr) : m_arr(arr) {
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
