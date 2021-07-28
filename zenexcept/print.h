#pragma once

#include <iostream>
#include <string_view>

namespace zpp {

static std::ostream &operator<<(std::ostream &os, source_location const &location) {
    os << location.file_name() << ":"
      << location.line() << ":"
      << location.column() << ": "
      << location.function_name() << "()";
    return os;
}

static void print(std::string_view const &message, ZPP_TRACEBACK) {
    std::cout << zpp_tb.location() << ": " << message << std::endl;
}

static void print_traceback(ZPP_TRACEBACK) {
    std::cout << "=== Begin ZPP Traceback ===" << std::endl;
    for (zpp::traceback const *p = &zpp_tb; p; p = p->previous()) {
        std::cout << "  " << p->location() << std::endl;
    }
    std::cout << "=== End of ZPP Traceback ===" << std::endl;
}

}
