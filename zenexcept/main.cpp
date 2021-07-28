#include <iostream>
#include <string_view>
#include "source_location.h"
#include "print.h"

template <class T>
void fun(T x) {
    zpp::log(x);
}

int main() {
    fun("Hello, world\n");
    *(int *)nullptr = 0;
    return 0;
}
