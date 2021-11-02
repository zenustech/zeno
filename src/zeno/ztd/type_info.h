#pragma once


#include <zeno/common.h>
#include <typeinfo>

#if defined(__GNUC__) || defined(__clang__)

#include <cstdlib>
#include <cxxabi.h>

ZENO_NAMESPACE_BEGIN
namespace ztd {
inline namespace _type_info_h {

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
}
ZENO_NAMESPACE_END

#else

ZENO_NAMESPACE_BEGIN
namespace ztd {
inline namespace _type_info_h {

inline std::string cpp_type_name(std::type_info const &type) {
    // MSVC is able to return demanged name directly via name()
    return type.name();
}

}
}
ZENO_NAMESPACE_END

#endif
