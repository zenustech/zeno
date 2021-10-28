#include <cstdio>
#include <tuple>

using namespace std;

template <class T>
void show(T &&) {
    printf("%s\n", __PRETTY_FUNCTION__);
}

template <class T>
void bar(T &&t) {
    printf("%s\n", __PRETTY_FUNCTION__);
    show(forward<T>(t));
    show(t);
    printf("===\n");
}

template <class T>
void foo(type_identity_t<T>) {
    printf("%s\n", __PRETTY_FUNCTION__);
}

int main() {
    const int a = 1;
    foo<const int &>(a);
    bar(a);
    int b = 1;
    foo<int &>(b);
    bar(b);
    int c = 1;
    foo<int &&>(move(c));
    bar(move(c));
    return 0;
}
