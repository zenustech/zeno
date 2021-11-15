#include <vector>

template <class T>
struct vector_view {
    T *m_base;
    size_t m_size;

    constexpr T *data() const {
        return m_base;
    }

    constexpr size_t size() const {
        return m_size;
    }
};

struct vector {
    std::vector<std::byte> m_data;

    template <class T>
    constexpr vector_view<T> view() {
        auto base = static_cast<T *>(m_data.data());
        auto size = (m_data.size() + sizeof(T) - 1) / sizeof(T);
        return {base, size};
    }

    template <class T>
    constexpr vector_view<T> view() const {
        const_cast<vector *>(this)->view<std::add_const_t<T>>();
    }
};


