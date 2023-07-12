// ======================================================================== //
// Copyright 2009-2017 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#define _CRT_SECURE_NO_WARNINGS

#include <cstddef>
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <cstring>
#include <stdint.h>

#ifdef _WIN32
  #include <intrin.h>
  #include <windows.h>
#endif

////////////////////////////////////////////////////////////////////////////////
/// Makros
////////////////////////////////////////////////////////////////////////////////

#ifdef _WIN32
#undef __noinline
#define __noinline             __declspec(noinline)
//#define __forceinline        __forceinline
//#define __restrict           __restrict
#ifdef __INTEL_COMPILER
#define __restrict__           __restrict
#else
#define __restrict__           //__restrict // causes issues with MSVC
#endif
#define __thread               __declspec(thread)
#define __aligned(...)           __declspec(align(__VA_ARGS__))
//#define __FUNCTION__           __FUNCTION__
#define debugbreak()           __debugbreak()

#else
#undef __noinline
#undef __forceinline
#define __noinline             __attribute__((noinline))
#define __forceinline          inline __attribute__((always_inline))
//#define __restrict             __restrict
//#define __thread               __thread
#define __aligned(...)           __attribute__((aligned(__VA_ARGS__)))
#define __FUNCTION__           __PRETTY_FUNCTION__
#define debugbreak()           asm ("int $3")
#endif

#ifdef __GNUC__
  #define MAYBE_UNUSED __attribute__((unused))
#else
  #define MAYBE_UNUSED
#endif

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
#define   likely(expr) (expr)
#define unlikely(expr) (expr)
#else
#define   likely(expr) __builtin_expect((bool)(expr),true )
#define unlikely(expr) __builtin_expect((bool)(expr),false)
#endif

////////////////////////////////////////////////////////////////////////////////
/// Error handling and debugging
////////////////////////////////////////////////////////////////////////////////

/* debug printing macros */
#define STRING(x) #x
#define TOSTRING(x) STRING(x)
#define CODE_LOCATION __FILE__ " (" TOSTRING(__LINE__) ")"
#define PING std::cout << CODE_LOCATION << ": " << __FUNCTION__ << std::endl
#define PRINT(x) std::cout << STRING(x) << " = " << (x) << std::endl
#define PRINT2(x,y) std::cout << STRING(x) << " = " << (x) << ", " << STRING(y) << " = " << (y) << std::endl
#define PRINT3(x,y,z) std::cout << STRING(x) << " = " << (x) << ", " << STRING(y) << " = " << (y) << ", " << STRING(z) << " = " << (z) << std::endl
#define PRINT4(x,y,z,w) std::cout << STRING(x) << " = " << (x) << ", " << STRING(y) << " = " << (y) << ", " << STRING(z) << " = " << (z) << ", " << STRING(w) << " = " << (w) << std::endl

#define THROW_RUNTIME_ERROR(str) \
  throw std::runtime_error(std::string(__FILE__) + " (" + std::to_string((long long)__LINE__) + "): " + std::string(str));
#define FATAL(x) THROW_RUNTIME_ERROR(x)
#define WARNING(x) std::cerr << "Warning:" << std::string(x) << std::endl

#define NOT_IMPLEMENTED FATAL(std::string(__FUNCTION__) + " not implemented")

// NOTE(jda) - These macros are used to construct the last UNUSED(...) macro,
//             used to mark a variable number of arguments as unused so the
//             compiler doesn't warn when -Wextra (gcc/clang/icc) is used. Only
//             works with 1 to 5 passed arguments.
#define UNUSED_1(x) (void)x
#define UNUSED_2(x, y) UNUSED_1(x); UNUSED_1(y)
#define UNUSED_3(x, ...) UNUSED_2(x, UNUSED_2(__VA_ARGS__))
#define UNUSED_4(x, ...) UNUSED_2(x, UNUSED_3(__VA_ARGS__))
#define UNUSED_5(x, ...) UNUSED_2(x, UNUSED_4(__VA_ARGS__))

// NUM_ARGS(...) evaluates to the literal number of the passed-in arguments.
#define _NUM_ARGS2(X,X5,X4,X3,X2,X1,N,...) N
#define NUM_ARGS(...) _NUM_ARGS2(0,__VA_ARGS__,5,4,3,2,1,0)

#define _UNUSED_N3(N, ...) UNUSED_##N(__VA_ARGS__)
#define _UNUSED_N2(N, ...) _UNUSED_N3(N, __VA_ARGS__)
#define UNUSED(...) _UNUSED_N2(NUM_ARGS(__VA_ARGS__), __VA_ARGS__)

#if defined(__x86_64__) || defined(__ia64__) || defined(_M_X64)
  #define __X86_64__
#endif

////////////////////////////////////////////////////////////////////////////////
/// Basic Types
////////////////////////////////////////////////////////////////////////////////

/* default floating-point type */
typedef float real;

/* windows does not have ssize_t */
#ifdef __WIN32
#ifdef __X86_64__
typedef int64_t ssize_t;
#else
typedef int32_t ssize_t;
#endif
#endif

////////////////////////////////////////////////////////////////////////////////
/// Disable some compiler warnings
////////////////////////////////////////////////////////////////////////////////

#if defined(__INTEL_COMPILER)
#pragma warning(disable:265 ) // floating-point operation result is out of range
#pragma warning(disable:383 ) // value copied to temporary, reference to temporary used
#pragma warning(disable:869 ) // parameter was never referenced
#pragma warning(disable:981 ) // operands are evaluated in unspecified order
#pragma warning(disable:1418) // external function definition with no prior declaration
#pragma warning(disable:1419) // external declaration in primary source file
#pragma warning(disable:1572) // floating-point equality and inequality comparisons are unreliable
#pragma warning(disable:94  ) // the size of an array must be greater than zero
#pragma warning(disable:1599) // declaration hides parameter
#pragma warning(disable:424 ) // extra ";" ignored
#pragma warning(disable:2196) // routine is both "inline" and "noinline"
#pragma warning(disable:177 ) // label was declared but never referenced
#pragma warning(disable:114 ) // function was referenced but not defined
#endif

#if defined(_MSC_VER)
#pragma warning(disable:4200) // nonstandard extension used : zero-sized array in struct/union
#pragma warning(disable:4800) // forcing value to bool 'true' or 'false' (performance warning)
#pragma warning(disable:4267) // '=' : conversion from 'size_t' to 'unsigned long', possible loss of data
#pragma warning(disable:4244) // 'argument' : conversion from 'ssize_t' to 'unsigned int', possible loss of data
#pragma warning(disable:4355) // 'this' : used in base member initializer list
#pragma warning(disable:391 ) // '<=' : signed / unsigned mismatch
#pragma warning(disable:4018) // '<' : signed / unsigned mismatch
#pragma warning(disable:4305) // 'initializing' : truncation from 'double' to 'float'
#pragma warning(disable:4068) // unknown pragma
#pragma warning(disable:4146) // unary minus operator applied to unsigned type, result still unsigned
#pragma warning(disable:4838) // conversion from 'unsigned int' to 'const int' requires a narrowing conversion)
#pragma warning(disable:4227) // anachronism used : qualifiers on reference are ignored
#endif

#if defined(__clang__) && !defined(__INTEL_COMPILER)
#pragma clang diagnostic ignored "-Wunknown-pragmas"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wreorder"
#pragma clang diagnostic ignored "-Wmicrosoft"
#pragma clang diagnostic ignored "-Wunused-private-field"
#pragma clang diagnostic ignored "-Wunused-local-typedef"
#pragma clang diagnostic ignored "-Wunused-function"
#endif
