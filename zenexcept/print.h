#pragma once

#include <iostream>
#include <string_view>

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
