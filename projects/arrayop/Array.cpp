#include <zeno/utils/allocator.h>
#include <type_traits>
#include <cstdint>
#include <vector>
#include <array>
#include <tuple>

namespace zeno {

template <class T>
struct vector_view {
    std::vector<std::byte> *m_origin;

    explicit vector_view(std::vector<std::byte> *origin)
        : m_origin(origin) {}

    T *data() const {
        return (T *)m_origin->data();
    }

    std::size_t size() const {
        return m_origin->size() / sizeof(T);
    }
};

enum class DataType : std::uint8_t {
    Float,
    Int,
};

struct vector_storage {
    static_assert(std::is_pod_v<std::byte>);
    mutable std::vector<std::byte, allocator<std::byte>> m_data;

    DataType m_dtype = DataType::Float;

    DataType dtype() const { return m_dtype; }

    template <class T>
    void assume_dtype() const { // todo
    }

    template <class T>
    vector_view<T> view() const {
        assume_dtype<T>();
        return vector_view<T>{&m_data};
    }
};

template <class ...Ts>
struct multi_vector_view {
    using type_tuple_t = std::tuple<Ts...>;
    inline static constexpr std::size_t kNumComps = sizeof...(Ts);

    std::array<std::vector<std::byte> *, kNumComps> m_origin;

    explicit multi_vector_view(vector_view<Ts> const &...views)
        : m_origin({views.m_origin...}) {
    }

    template <std::size_t I>
    std::tuple_element_t<I, type_tuple_t> *data() const {
        std::get<I>(m_origin)->data();
    }

    template <std::size_t ...Is>
    std::size_t _helper_size(std::index_sequence<Is...>) const {
        return std::min({std::get<Is>()...});
    }

    std::size_t size() const {
        return _helper_size(std::make_index_sequence<kNumComps>{});
    }
};

template <class ...Ts>
multi_vector_view(vector_view<Ts> const &...) -> multi_vector_view<Ts...>;

}
