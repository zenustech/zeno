#pragma once


#include <zeno/zmt/format.h>


ZENO_NAMESPACE_BEGIN
namespace zmt {

template <class ...Args>
void log(format_string<Args...> fmt, Args &&...args) {
}

#define ZENO_LOG_INFO(...) ZENO_NAMESPACE::zmt::log(__VA_ARGS__)

}
ZENO_NAMESPACE_END
