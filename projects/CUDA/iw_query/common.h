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

#include "platform.h"
// std
// #include <mutex>
// #include <stdexcept>
// #include <algorithm> // std::min etc on windows

#ifdef _WIN32
// ----------- windows only -----------
typedef unsigned long long id_t;
# ifndef _USE_MATH_DEFINES
#   define _USE_MATH_DEFINES
# endif
# include <cmath>
# ifdef _M_X64
typedef long long ssize_t;
# else
typedef int ssize_t;
# endif
#else
// ----------- NOT windows -----------
# include "unistd.h"
#endif

#include "math.h"

#include <stdint.h>

#ifdef _WIN32
#  ifdef ospray_common_EXPORTS
#    define OSPCOMMON_INTERFACE __declspec(dllexport)
#  else
#    define OSPCOMMON_INTERFACE __declspec(dllimport)
#  endif
#else
#  define OSPCOMMON_INTERFACE
#endif

#ifdef NDEBUG
# define Assert(expr) /* nothing */
# define Assert2(expr,expl) /* nothing */
# define AssertError(errMsg) /* nothing */
#else
# define Assert(expr)                                                   \
  ((void)((expr) ? 0 : ((void)ospcommon::doAssertion(__FILE__, __LINE__, #expr, nullptr), 0)))
# define Assert2(expr,expl)                                             \
  ((void)((expr) ? 0 : ((void)ospcommon::doAssertion(__FILE__, __LINE__, #expr, expl), 0)))
# define AssertError(errMsg)                            \
  doAssertion(__FILE__,__LINE__, (errMsg), nullptr)
#endif

#ifdef _WIN32
#define __PRETTY_FUNCTION__ __FUNCSIG__
#endif
#define NOTIMPLEMENTED    throw std::runtime_error(std::string(__PRETTY_FUNCTION__)+": not implemented...");

#define SCOPED_LOCK(x) \
  std::lock_guard<std::mutex> _lock(x); \
  (void)_lock;

namespace ospcommon {

  typedef unsigned char byte_t;

  /*! return system time in seconds */
  OSPCOMMON_INTERFACE double getSysTime();
  OSPCOMMON_INTERFACE void doAssertion(const char *file, int line, const char *expr, const char *expl);
  /*! remove specified num arguments from an ac/av arglist */
  OSPCOMMON_INTERFACE void removeArgs(int &ac, char **&av, int where, int howMany);
  OSPCOMMON_INTERFACE void loadLibrary(const std::string &name);
  OSPCOMMON_INTERFACE void *getSymbol(const std::string &name);


#ifdef _WIN32
#  define osp_snprintf sprintf_s
#else
#  define osp_snprintf snprintf
#endif
  
  /*! added pretty-print function for large numbers, printing 10000000 as "10M" instead */
  inline std::string prettyDouble(const double val) {
    const double absVal = abs(val);
    char result[1000];

    if      (absVal >= 1e+15f) osp_snprintf(result,1000,"%.1f%c",val/1e18f,'E');
    else if (absVal >= 1e+15f) osp_snprintf(result,1000,"%.1f%c",val/1e15f,'P');
    else if (absVal >= 1e+12f) osp_snprintf(result,1000,"%.1f%c",val/1e12f,'T');
    else if (absVal >= 1e+09f) osp_snprintf(result,1000,"%.1f%c",val/1e09f,'G');
    else if (absVal >= 1e+06f) osp_snprintf(result,1000,"%.1f%c",val/1e06f,'M');
    else if (absVal >= 1e+03f) osp_snprintf(result,1000,"%.1f%c",val/1e03f,'k');
    else if (absVal <= 1e-12f) osp_snprintf(result,1000,"%.1f%c",val*1e15f,'f');
    else if (absVal <= 1e-09f) osp_snprintf(result,1000,"%.1f%c",val*1e12f,'p');
    else if (absVal <= 1e-06f) osp_snprintf(result,1000,"%.1f%c",val*1e09f,'n');
    else if (absVal <= 1e-03f) osp_snprintf(result,1000,"%.1f%c",val*1e06f,'u');
    else if (absVal <= 1e-00f) osp_snprintf(result,1000,"%.1f%c",val*1e03f,'m');
    else osp_snprintf(result,1000,"%f",(float)val);
    return result;
  }

  /*! added pretty-print function for large numbers, printing 10000000 as "10M" instead */
  inline std::string prettyNumber(const size_t s) {
    const double val = s;
    char result[1000];

    if      (val >= 1e+15f) osp_snprintf(result,1000,"%.1f%c",val/1e18f,'E');
    else if (val >= 1e+15f) osp_snprintf(result,1000,"%.1f%c",val/1e15f,'P');
    else if (val >= 1e+12f) osp_snprintf(result,1000,"%.1f%c",val/1e12f,'T');
    else if (val >= 1e+09f) osp_snprintf(result,1000,"%.1f%c",val/1e09f,'G');
    else if (val >= 1e+06f) osp_snprintf(result,1000,"%.1f%c",val/1e06f,'M');
    else if (val >= 1e+03f) osp_snprintf(result,1000,"%.1f%c",val/1e03f,'k');
    else osp_snprintf(result,1000,"%lu",s);
    return result;
  }
#undef osp_snprintf

  template <typename T>
  inline std::pair<bool, T> getEnvVar(const std::string &/*var*/)
  {
    static_assert(!std::is_same<T, float>::value &&
                  !std::is_same<T, int>::value &&
                  !std::is_same<T, std::string>::value,
                  "You can only get an int, float, or std::string "
                  "when using ospray::getEnvVar<T>()!");
    return {false, {}};
  }

} // ::ospcommon
