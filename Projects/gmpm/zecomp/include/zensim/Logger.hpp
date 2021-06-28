/// reference: taichi/common/core.h
#pragma once

#include "zensim/Singleton.h"
#include "zensim/tpls/fmt/core.h"
#include "zensim/tpls/spdlog/spdlog.h"
#include "zensim/tpls/spdlog/sinks/basic_file_sink.h"

namespace zs {

// Reference:
// https://blog.kowalczyk.info/article/j/guide-to-predefined-macros-in-c-compilers-gcc-clang-msvc-etc..html

///
#define SPDLOG(option, ...)                                                   \
  spdlog::option(fmt::format("[{}:{}@{}] ", __FILE__, __FUNCTION__, __LINE__) \
                 + fmt::format(__VA_ARGS__))

#define ZS_TRACE(...) SPDLOG(trace, __VA_ARGS__)
#define ZS_DEBUG(...) SPDLOG(debug, __VA_ARGS__)
#define ZS_INFO(...) SPDLOG(info, __VA_ARGS__)
#define ZS_WARN(...) SPDLOG(warn, __VA_ARGS__)
#define ZS_ERROR(...)           \
  {                             \
    SPDLOG(error, __VA_ARGS__); \
    ZS_UNREACHABLE;             \
  }
#define ZS_CRITICAL(...)           \
  {                                \
    SPDLOG(critical, __VA_ARGS__); \
    ZS_UNREACHABLE;                \
  }
/// conditional
#define ZS_ERROR_IF(condition, ...) \
  if (condition) {                  \
    ZS_ERROR(__VA_ARGS__);          \
  }
#define ZS_WARN_IF(condition, ...) \
  if (condition) {                 \
    ZS_WARN(__VA_ARGS__);          \
  }

  struct Logger : Singleton<Logger> {
    ;
  };

}  // namespace zs
