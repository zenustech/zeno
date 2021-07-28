#include <iostream>
#include <string_view>
#include "source_location.h"

namespace zpp {

void log(const std::string_view message,
         const source_location &location = source_location::current())
{
    std::cout << "file: "
              << location.file_name() << "("
              << location.line() << ":"
              << location.column() << ") `"
              << location.function_name() << "`: "
              << message << '\n';
}

}

template <class T>
void fun(T x) {
    zpp::log(x);
}

int main() {
    fun("Hello, world\n");
    return 0;
}
