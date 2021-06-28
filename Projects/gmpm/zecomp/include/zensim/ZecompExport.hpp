#pragma once

// https://abseil.io/docs/cpp/platforms/macros

#ifndef ZECOMP_EXPORT
#  if defined(_MSC_VER) || defined(__CYGWIN__)
#    ifdef ZeComp_EXPORT
#      define ZECOMP_EXPORT __declspec(dllexport)
#    else
#      define ZECOMP_EXPORT __declspec(dllimport)
#    endif
#  elif defined(__clang__) || defined(__GNUC__)
#    define ZECOMP_EXPORT __attribute__((visibility("default")))
#  endif
#endif
