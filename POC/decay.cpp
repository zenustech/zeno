#include <cstdio>
#include <tuple>
#include <variant>
#include <vector>

using namespace std;

template <class T>
void show(T &&) {
    printf("%s\n", __PRETTY_FUNCTION__);
}

template <class T>
void show() {
    printf("%s\n", __PRETTY_FUNCTION__);
}

//template <int I>
struct C {
    C() {
        printf("%s\n", __PRETTY_FUNCTION__);
    }
    ~C() {
        printf("%s\n", __PRETTY_FUNCTION__);
    }
    C(C const &) {
        printf("%s\n", __PRETTY_FUNCTION__);
    }
    C &operator=(C const &) {
        printf("%s\n", __PRETTY_FUNCTION__);
        return *this;
    }
    C(C &&) {
        printf("%s\n", __PRETTY_FUNCTION__);
    }
    C &operator=(C &&) {
        printf("%s\n", __PRETTY_FUNCTION__);
        return *this;
    }
};

void foo(auto &&t) {
    show(forward<decltype(t)>(t));
}

template <class T>
void bar(T &&t) {
    show(forward<T>(t));
}

int main() {
    foo(3);
    bar(3);
    return 0;
}
