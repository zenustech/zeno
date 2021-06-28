#pragma once
// from taichi/common/core.h

// Windows
#if defined(_WIN64)
#  define ZS_PLATFORM_WINDOWS
#endif

#if defined(_WIN32) && !defined(_WIN64)
static_assert(false, "32-bit Windows systems are not supported")
#endif

// Linux
#if defined(__linux__)
#  define ZS_PLATFORM_LINUX
#endif

// OSX
#if defined(__APPLE__)
#  define ZS_PLATFORM_OSX
#endif

// Unix
#if (defined(ZS_PLATFORM_LINUX) || defined(ZS_PLATFORM_OSX))
#  define ZS_PLATFORM_UNIX
#endif

#if defined(ZS_PLATFORM_WINDOWS)
#  define ZS_UNREACHABLE __assume(0);
#else
#  define ZS_UNREACHABLE __builtin_unreachable();
#endif
