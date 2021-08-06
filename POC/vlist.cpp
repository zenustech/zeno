#include <variant>
#include <typeinfo>
#include <cstdio>


template <class...>
struct vlist {
};

template <class T, class ...Ts>
struct vlist<T, Ts...> {
    using head = T;
    using rest = vlist<Ts...>;
};

template <template <class> class Tp, class L>
struct vlistfor {
    void operator()() {
        Tp<typename L::head>()();
        vlistfor<Tp, typename L::rest>()();
    }
};

template <template <class> class Tp>
struct vlistfor<Tp, vlist<>> {
    void operator()() {
    }
};

template <class T>
struct myfunc {
    void operator()() {
        printf("%s\n", typeid(T).name());
    }
};

int main(void) {
    vlistfor<myfunc, vlist<int, float, double>>()();
}
