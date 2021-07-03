#include <vector>
#include <type_traits>
#include <magic_enum.hpp>
#include <iostream>

template <int First, int Last, typename Lambda>
inline void static_for(Lambda const& f) {
    if constexpr (First < Last) {
        if (f(std::integral_constant<int, First>{})) {
            static_for<First + 1, Last>(f);
        }
    }
}

using std::cout;
using std::endl;

enum class dtype : int {
    none, i32, f32,
};

inline constexpr size_t dtype_count = magic_enum::enum_values<dtype>().size();

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

template <dtype dt>
using dtype_t = typename dtype_traits<dt>::type;

struct array {
    std::vector<char> m_data;
    dtype m_type;

    constexpr dtype type() const { return m_type; }
    void *data() const { return (void *)m_data.data(); }
};

template <class T>
void do_apply(void *p) {
    printf("%s\n", typeid(T).name());
}

void apply(array &a) {
    static_for<0, magic_enum::enum_values<dtype>().size()>([&](auto i) {
        constexpr dtype dt = magic_enum::enum_cast<dtype>(i).value();
        if (a.type() == dt) {
            do_apply<dtype_t<dt>>(a.data());
            return false;
        }
        return true;
    });
}

int main(void)
{
    array a;
    a.m_type = dtype::i32;
    apply(a);
    return 0;
}
