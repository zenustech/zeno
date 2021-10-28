#pragma once


#include <zeno/common.h>


ZENO_NAMESPACE_BEGIN
namespace zmt {

template <class ...Args>
using format_string = std::string_view;

template <class ...Args>
inline void format(format_string<Args...> fmt, Args &&...args) {
}

}
ZENO_NAMESPACE_END
