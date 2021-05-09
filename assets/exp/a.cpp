#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <iostream>

using std::cout;
using std::endl;



template <class T, class S>
struct is_promotable {
    static constexpr bool value = false;
    using type = void;
};

template <class T, size_t N>
struct is_promotable<Vec<N, T>, Vec<N, T>> {
    static constexpr bool value = true;
    using type = Vec<N, T>;
};

template <class T, size_t N>
struct is_promotable<Vec<N, T>, T> {
    static constexpr bool value = true;
    using type = Vec<N, T>;
};

template <class T, size_t N>
struct is_promotable<T, Vec<N, T>> {
    static constexpr bool value = true;
    using type = Vec<N, T>;
};

template <class T>
struct is_promotable<T, T> {
    static constexpr bool value = true;
    using type = T;
};

template <class T, class S>
inline constexpr bool is_promotable_v = is_promotable<std::decay_t<T>, std::decay_t<S>>::value;

template <class T, class S>
using is_promotable_t = typename is_promotable<std::decay_t<T>, std::decay_t<S>>::type;


int main(void) {
    cout << is_promotable_v<Vec<3, float>, float> << endl;
}
