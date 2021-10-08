#pragma once


#include <typeinfo>

#if defined(__GNUC__) || defined(__clang__)

#include <cstdlib>
#include <cxxabi.h>

namespace zeno2::ztd {

inline std::string cpp_demangle(const char *name) {
    int status;
    char *p = abi::__cxa_demangle(name, 0, 0, &status);
    std::string s = p;
    std::free(p);
    return s;
}

inline std::string cpp_type_name(std::type_info const &type) {
    return cpp_demangle(type.name());
}

}

#else

namespace zeno2::ztd {

inline std::string cpp_type_name(std::type_info const &type) {
    // MSVC is able to return demanged name directly via name()
    return type.name();
}

}

#endif
