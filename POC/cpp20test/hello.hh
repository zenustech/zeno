module;

#include <array>

export module hello;

export void hello();

export template <class T>
auto get_array() {
    return std::array<T, 1>();
}
