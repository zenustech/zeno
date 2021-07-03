#include <vector>
#include <type_traits>
#include <magic_enum.hpp>
#include <iostream>

template <int First, int Last, typename Lambda>
inline constexpr bool static_for(Lambda const &f) {
    if constexpr (First < Last) {
        if (f(std::integral_constant<int, First>{})) {
            return true;
        } else {
            return static_for<First + 1, Last>(f);
        }
    }
    return false;
}

using std::cout;
using std::endl;

enum class dtype : int {
    none, i32, f32,
};

template <dtype dt>
struct dtype_traits {
};

template <>
struct dtype_traits<dtype::none> {
    using type = void;
};

template <>
struct dtype_traits<dtype::i32> {
    using type = int;
};

template <>
struct dtype_traits<dtype::f32> {
    using type = float;
};

constexpr auto dtype_name(dtype dt) {
    return magic_enum::enum_name(dt);
}

constexpr size_t dtype_size(dtype dt) {
    size_t ret = 0;
    static_for<0, magic_enum::enum_values<dtype>().size()>([&](auto i) {
        constexpr auto t = magic_enum::enum_cast<dtype>(i).value();
        if (dt == t) {
            using T = typename dtype_traits<t>::type;
            if constexpr (std::is_void_v<T>)
                ret = 0;
            else
                ret = sizeof(T);
            return true;
        }
        return false;
    });
    return ret;
}

struct array {
    std::vector<char> m_data;
    dtype m_type;
    size_t m_size;

    constexpr dtype type() const { return m_type; }
    void *data() const { return (void *)m_data.data(); }
    size_t size() const { return m_size; }
};

template <class T>
void do_apply(T *p, size_t n) {
    printf("%s\n", typeid(T).name());
}

void apply(array &a) {
    auto dt = a.type();
    static_for<0, magic_enum::enum_values<dtype>().size()>([&](auto i) {
        constexpr auto t = magic_enum::enum_cast<dtype>(i).value();
        if (dt == t) {
            using T = typename dtype_traits<t>::type;
            do_apply<T>((T *)a.data(), a.size());
            return true;
        }
        return false;
    });
}

int main(void)
{
    array a;
    a.m_type = dtype::i32;
    apply(a);
    return 0;
}
