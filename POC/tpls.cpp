#include <iostream>

using std::cout;
using std::endl;


struct MC {
    int operator+(int x) const {
        return x + 1;
    }
};

struct op_add {
    template <class T1, class T2, decltype(
        std::declval<T1>() + std::declval<T2>()
        , true) = true>
    static auto apply(T1 const &x, T2 const &y) {
        return x + y;
    }
};

template <class T, class ...>
using _left_t = T;

template <class Op, class ...Ts>
struct is_op_valid : std::false_type {
    void apply(Ts const &...ts) {
    }
};

template <class Op, class ...Ts>
struct is_op_valid<_left_t<Op, decltype(
    Op::apply(std::declval<Ts>()...))>, Ts...> : std::true_type {

    auto apply(Ts const &...ts) {
        return Op::apply(std::forward<Ts>(ts)...);
    }
};

int main()
{
    cout << is_op_valid<op_add, MC, MC>::value << endl;
    cout << is_op_valid<op_add, MC, int>::value << endl;
    cout << is_op_valid<op_add, int, int>::value << endl;
    cout << is_op_valid<op_add, int>::value << endl;
}
