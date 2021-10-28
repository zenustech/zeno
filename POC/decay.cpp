#include <cstdio>
#include <tuple>

using namespace std;

template <class T>
void show() {
    printf("%s\n", __PRETTY_FUNCTION__);
}

int main() {
    {
        using T = int &;
        using S = T &;
        show<S>();
    }
    {
        using T = int &;
        using S = T &&;
        show<S>();
    }
    {
        using T = int &&;
        using S = T &;
        show<S>();
    }
    {
        using T = int &&;
        using S = T &&;
        show<S>();
    }
    {
        using T = int const &;
        using S = T &;
        show<S>();
    }
    {
        using T = int const &;
        using S = T &&;
        show<S>();
    }
    {
        using T = int const &&;
        using S = T &;
        show<S>();
    }
    {
        using T = int const &&;
        using S = T &&;
        show<S>();
    }
    {
        using T = int &;
        using S = T const &;
        show<S>();
    }
    {
        using T = int &;
        using S = T const &&;
        show<S>();
    }
    {
        using T = int &&;
        using S = T const &;
        show<S>();
    }
    {
        using T = int &&;
        using S = T const &&;
        show<S>();
    }
    return 0;
}
