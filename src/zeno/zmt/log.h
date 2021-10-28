#pragma once


#include <zeno/zmt/format.h>


ZENO_NAMESPACE_BEGIN
namespace zmt {

enum class log_level {
    trace = 0,
    debug = 1,
    info = 2,
    warn = 3,
    crit = 4,
    error = 5,
    fatal = 6,
};

void set_log_level(log_level lev);
void output_log(log_level lev, std::string_view msg);

#define ZENO_LOG(lev, ...) ZENO_NAMESPACE::zmt::output_log(lev, ZENO_NAMESPACE::zmt::format(__VA_ARGS__))
#define ZENO_LOG_TRACE(...) ZENO_LOG(ZENO_NAMESPACE::zmt::log_level::trace, __VA_ARGS__)
#define ZENO_LOG_DEBUG(...) ZENO_LOG(ZENO_NAMESPACE::zmt::log_level::debug, __VA_ARGS__)
#define ZENO_LOG_INFO(...) ZENO_LOG(ZENO_NAMESPACE::zmt::log_level::info, __VA_ARGS__)
#define ZENO_LOG_WARN(...) ZENO_LOG(ZENO_NAMESPACE::zmt::log_level::warn, __VA_ARGS__)
#define ZENO_LOG_CRIT(...) ZENO_LOG(ZENO_NAMESPACE::zmt::log_level::crit, __VA_ARGS__)
#define ZENO_LOG_ERROR(...) ZENO_LOG(ZENO_NAMESPACE::zmt::log_level::error, __VA_ARGS__)
#define ZENO_LOG_FATAL(...) ZENO_LOG(ZENO_NAMESPACE::zmt::log_level::fatal, __VA_ARGS__)

}
ZENO_NAMESPACE_END
