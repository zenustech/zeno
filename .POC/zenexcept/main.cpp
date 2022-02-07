#include <iostream>
#include <string_view>
#include "exception.h"
#include "source_location.h"
#include "print.h"

void bar(int a, ZPP_TRACEBACK) {
    printf("a=%d\n", a);
    throw zpp::exception();
}

template <class T>
void foo(T x, int y = 0, ZPP_TRACEBACK) {
    print_traceback(zpp_tb);
}

int main() {
    foo("Hello, world\n");
    return 0;
}
