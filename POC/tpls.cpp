#include <iostream>

using namespace std;


struct MC {};

template <class, class = void>
struct has_add_one : false_type {
};

template <class T>
struct has_add_one<T, std::void_t<decltype(declval<T>() + 1)>> : true_type {
};

int main()
{
    cout << has_add_one<MC>::value << endl;
}
