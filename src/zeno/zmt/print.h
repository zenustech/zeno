#pragma once


#include <zeno/common.h>
#include <zeno/zmt/format.h>
#include <iostream>


ZENO_NAMESPACE_BEGIN
namespace zmt {

template <class ...Args>
inline void print(std::string_view fmt, Args &&...args) {
    format_to(std::cout, fmt, std::forward<Args>(args)...);
}

template <class ...Args>
inline void eprint(std::string_view fmt, Args &&...args) {
    format_to(std::cerr, fmt, std::forward<Args>(args)...);
}

template <class ...Args>
inline void println(std::string_view fmt, Args &&...args) {
    print(std::move(fmt), std::forward<Args>(args)...);
    std::cout << std::endl;
}

template <class ...Args>
inline void eprintln(std::string_view fmt, Args &&...args) {
    eprint(std::move(fmt), std::forward<Args>(args)...);
    std::cerr << std::endl;
}

}
ZENO_NAMESPACE_END
