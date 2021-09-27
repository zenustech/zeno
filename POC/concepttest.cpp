#include <iostream>
#include <concepts>
#include <type_traits>

template <class T>
struct Point {
    T x, y;
};

template <class T>
concept Add = requires(T a, T b) {
    a + b;
};

template <class T>
concept Mul = requires(T a, T b) {
    a + b;
};

template <class T>
concept Copy = requires(T const &a) {
    T(a);
};

template <class T>
requires Add<T> && Mul<T> && Copy<T>
auto length(Point<T> const &self) {
    return self.x * self.x + self.y * self.y;
}

int main() {
    Point<int> p{1, 4};
    std::cout << length(p) << std::endl;
}
