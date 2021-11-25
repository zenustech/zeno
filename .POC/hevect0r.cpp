#include <vector>
#include <string>
#include <typeindex>

template <class T>
struct span {
    T *m_base;
    std::size_t m_size;

    constexpr T *data() const {
        return m_base;
    }

    constexpr std::size_t size() const {
        return m_size;
    }

    constexpr T &operator[](std::size_t i) const {
        return m_base[i];
    }
};

struct vector {
    std::vector<std::byte> m_data;

    struct ChannelInfo {
        std::type_index type;
        std::size_t size;
        std::size_t count;
        std::string name;
    };

    std::vector<ChannelInfo> m_channels;

    template <class T>
    constexpr span<T> view(std::size_t ch) {
        auto base = reinterpret_cast<T *>(m_data.data());
        auto size = (m_data.size() + sizeof(T) - 1) / sizeof(T);
        return {base, size};
    }

    template <class T>
    constexpr span<T> view(std::size_t ch) const {
        return const_cast<vector *>(this)->view<std::add_const_t<T>>(ch);
    }
};
