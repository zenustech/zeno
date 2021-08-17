#ifdef ZENO_FAULTHANDLER
// https://github.com/taichi-dev/taichi/blob/eb769ebfc0cb6b48649a3aed8ccd293cbd4eb5ed/taichi/system/traceback.cpp
/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#define FMT_HEADER_ONLY
#include <vector>
#include <iostream>
#include <string>
#include <sstream>
#include <cstdio>
#include <algorithm>
#include <memory>
#include <mutex>
#include <fmt/core.h>
#include <fmt/color.h>

#ifdef __APPLE__
#include <execinfo.h>
#include <cxxabi.h>
#endif
#ifdef _WIN32
#include <windows.h>
// Never directly include <windows.h>. That will bring you evil max/min macros.
#if defined(min)
#undef min
#endif
#if defined(max)
#undef max
#endif
#include <intrin.h>
#include <dbghelp.h>

#pragma comment(lib, "dbghelp.lib")
//  https://gist.github.com/rioki/85ca8295d51a5e0b7c56e5005b0ba8b4
//
//  Debug Helpers
//
// Copyright (c) 2015 - 2017 Sean Farrell <sean.farrell@rioki.org>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//

namespace {
namespace dbg {
void trace(const char *msg, ...) {
  char buff[1024];

  va_list args;
  va_start(args, msg);
  vsnprintf(buff, 1024, msg, args);

  OutputDebugStringA(buff);

  va_end(args);
}

std::string basename(const std::string &file) {
  unsigned int i = file.find_last_of("\\/");
  if (i == std::string::npos) {
    return file;
  } else {
    return file.substr(i + 1);
  }
}

struct StackFrame {
  DWORD64 address;
  std::string name;
  std::string module;
  unsigned int line;
  std::string file;
};

inline std::vector<StackFrame> stack_trace() {
#if _WIN32
  DWORD machine = IMAGE_FILE_MACHINE_AMD64;
#else
  DWORD machine = IMAGE_FILE_MACHINE_I386;
#endif
  HANDLE process = GetCurrentProcess();
  HANDLE thread = GetCurrentThread();

  if (SymInitialize(process, NULL, TRUE) == FALSE) {
    trace(__FUNCTION__ ": Failed to call SymInitialize.");
    return std::vector<StackFrame>();
  }

  SymSetOptions(SYMOPT_LOAD_LINES);

  CONTEXT context = {};
  context.ContextFlags = CONTEXT_FULL;
  RtlCaptureContext(&context);

#if _WIN32
  STACKFRAME frame = {};
  frame.AddrPC.Offset = context.Rip;
  frame.AddrPC.Mode = AddrModeFlat;
  frame.AddrFrame.Offset = context.Rbp;
  frame.AddrFrame.Mode = AddrModeFlat;
  frame.AddrStack.Offset = context.Rsp;
  frame.AddrStack.Mode = AddrModeFlat;
#else
  STACKFRAME frame = {};
  frame.AddrPC.Offset = context.Eip;
  frame.AddrPC.Mode = AddrModeFlat;
  frame.AddrFrame.Offset = context.Ebp;
  frame.AddrFrame.Mode = AddrModeFlat;
  frame.AddrStack.Offset = context.Esp;
  frame.AddrStack.Mode = AddrModeFlat;
#endif

  bool first = true;

  std::vector<StackFrame> frames;
  while (StackWalk(machine, process, thread, &frame, &context, NULL,
                   SymFunctionTableAccess, SymGetModuleBase, NULL)) {
    StackFrame f = {};
    f.address = frame.AddrPC.Offset;

#if _WIN32
    DWORD64 moduleBase = 0;
#else
    DWORD moduleBase = 0;
#endif

    moduleBase = SymGetModuleBase(process, frame.AddrPC.Offset);

    char moduelBuff[MAX_PATH];
    if (moduleBase &&
        GetModuleFileNameA((HINSTANCE)moduleBase, moduelBuff, MAX_PATH)) {
      f.module = basename(moduelBuff);
    } else {
      f.module = "Unknown Module";
    }
#if _WIN32
    DWORD64 offset = 0;
#else
    DWORD offset = 0;
#endif
    char symbolBuffer[sizeof(IMAGEHLP_SYMBOL) + 255];
    PIMAGEHLP_SYMBOL symbol = (PIMAGEHLP_SYMBOL)symbolBuffer;
    symbol->SizeOfStruct = (sizeof IMAGEHLP_SYMBOL) + 255;
    symbol->MaxNameLength = 254;

    if (SymGetSymFromAddr(process, frame.AddrPC.Offset, &offset, symbol)) {
      f.name = symbol->Name;
    } else {
      DWORD error = GetLastError();
      trace(__FUNCTION__ ": Failed to resolve address 0x%X: %u\n",
            frame.AddrPC.Offset, error);
      f.name = "Unknown Function";
    }

    IMAGEHLP_LINE line;
    line.SizeOfStruct = sizeof(IMAGEHLP_LINE);

    DWORD offset_ln = 0;
    if (SymGetLineFromAddr(process, frame.AddrPC.Offset, &offset_ln, &line)) {
      f.file = line.FileName;
      f.line = line.LineNumber;
    } else {
      DWORD error = GetLastError();
      trace(__FUNCTION__ ": Failed to resolve line for 0x%X: %u\n",
            frame.AddrPC.Offset, error);
      f.line = 0;
    }

    if (!first) {
      frames.push_back(f);
    }
    first = false;
  }

  SymCleanup(process);

  return frames;
}
}  // namespace dbg
}
#endif
#ifdef __linux__
#include <execinfo.h>
#include <signal.h>
#include <ucontext.h>
#include <unistd.h>
#include <cxxabi.h>

namespace zeno::dbg {
static std::string calc_addr_to_line(
        std::string const &file, std::string const &name,
        std::string const &offset) {

    char cmd[2048];
    int ch;
    sprintf(cmd, "nm -p --defined-only '%s' 2>&1 | grep 'T %s$'", file.c_str(), name.c_str());
    FILE *fp = popen(cmd, "r");
    if (!fp) return "";
    std::string addr;
    while (1) {
        ch = fgetc(fp);
        if (ch == EOF)
            break;
        if (ch != ' ')
            addr += (char)ch;
    }
    pclose(fp);
    if (!addr.size()) return "";

    sprintf(cmd, "addr2line -e '%s' -- '%s'", file.c_str(), addr.c_str());
    fp = popen(cmd, "r");
    if (!fp) return "";
    std::string res;
    while (1) {
        ch = fgetc(fp);
        if (ch == EOF)
            break;
        if (ch != '\n')
            res += (char)ch;
    }
    pclose(fp);
    return res;
}
}
#endif

namespace zeno {
void print_traceback(int skip) {
#ifdef __APPLE__
  static std::mutex traceback_printer_mutex;
  // Modified based on
  // http://www.nullptr.me/2013/04/14/generating-stack-trace-on-os-x/
  // TODO: print line number instead of offset
  // (https://stackoverflow.com/questions/8278691/how-to-fix-backtrace-line-number-error-in-c)

  // record stack trace upto 128 frames
  void *callstack[128] = {};
  // collect stack frames
  int frames = backtrace((void **)callstack, 128);
  // get the human-readable symbols (mangled)
  char **strs = backtrace_symbols((void **)callstack, frames);
  std::vector<std::string> stack_frames;
  for (int i = 0; i < frames; i++) {
    char function_symbol[1024] = {};
    char module_name[1024] = {};
    int offset = 0;
    char addr[48] = {};
    // split the string, take out chunks out of stack trace
    // we are primarily interested in module, function and address
    sscanf(strs[i], "%*s %s %s %s %*s %d", module_name, addr, function_symbol,
           &offset);

    int valid_cpp_name = 0;
    //  if this is a C++ library, symbol will be demangled
    //  on success function returns 0
    char *function_name =
        abi::__cxa_demangle(function_symbol, NULL, 0, &valid_cpp_name);

    char stack_frame[4096] = {};
    sprintf(stack_frame, "* %28s | %7d | %s", module_name, offset,
            function_name);
    if (function_name != nullptr)
      free(function_name);

    std::string frameStr(stack_frame);
    stack_frames.push_back(frameStr);
  }
  free(strs);

  // Pretty print the traceback table
  // Exclude this function itself
  stack_frames.erase(stack_frames.begin());
  std::reverse(stack_frames.begin(), stack_frames.end());
  std::lock_guard<std::mutex> guard(traceback_printer_mutex);
  printf("\n");
  printf(
      "                            * Zeno - Stack Traceback *                  "
      "                  \n");
  printf(
      "========================================================================"
      "==================\n");
  printf(
      "|                       Module |  Offset | Function                     "
      "                 |\n");
  printf(
      "|-----------------------------------------------------------------------"
      "-----------------|\n");
  std::reverse(stack_frames.begin(), stack_frames.end());
  for (auto trace : stack_frames) {
    const int function_start = 39;
    const int line_width = 86;
    const int function_width = line_width - function_start - 2;
    int i;
    for (i = skip; i < (int)trace.size(); i++) {
      std::cout << trace[i];
      if (i > function_start + 3 &&
          (i - 3 - function_start) % function_width == 0) {
        std::cout << " |" << std::endl << " ";
        for (int j = 0; j < function_start; j++) {
          std::cout << " ";
        }
        std::cout << " | ";
      }
    }
    for (int j = 0;
         j < function_width + 2 - (i - 3 - function_start) % function_width;
         j++) {
      std::cout << " ";
    }
    std::cout << "|" << std::endl;
  }
  printf(
      "========================================================================"
      "==================\n");
  printf("\n");
#elif defined(_WIN32)
  // Windows
  fmt::print(fg(fmt::color::magenta), "************************\n");
  fmt::print(fg(fmt::color::magenta), "* Zeno Stack Traceback *\n");
  fmt::print(fg(fmt::color::magenta), "************************\n");

  std::vector<dbg::StackFrame> stack = dbg::stack_trace();
  for (unsigned int i = skip; i < stack.size(); i++) {
    fmt::print(fg(fmt::color::gray), "0x{:x}: ", stack[i].address);
    fmt::print(fg(fmt::color::yellow), stack[i].name);
    if (stack[i].file != std::string("")) {
      fmt::print(fg(fmt::color::gray), " at ");
      fmt::print(fg(fmt::color::cyan), "{}:{}", stack[i].file, stack[i].line);
    }
    fmt::print(fg(fmt::color::gray), " in ");
    fmt::print(fg(fmt::color::magenta), "{}\n", stack[i].module);
  }
#elif defined(__linux__)
  // Based on http://man7.org/linux/man-pages/man3/backtrace.3.html
  constexpr int BT_BUF_SIZE = 1024;
  int nptrs;
  void *buffer[BT_BUF_SIZE];
  char **strings;

  nptrs = backtrace(buffer, BT_BUF_SIZE);
  strings = backtrace_symbols(buffer, nptrs);

  if (strings == nullptr) {
    perror("backtrace_symbols");
    return;
  }

  fmt::print(fg(fmt::color::magenta), "************************\n");
  fmt::print(fg(fmt::color::magenta), "* Zeno Stack Traceback *\n");
  fmt::print(fg(fmt::color::magenta), "************************\n");

  // j = 0: zeno::print_traceback
  for (int j = 1 + skip; j < nptrs; j++) {
    std::string s(strings[j]);
    std::size_t slash = s.find("/");
    std::size_t bra = s.find("(");
    std::size_t ket = s.find(")");
    std::size_t ebra = s.find("[");
    std::size_t eket = s.find("]");
    std::size_t pls = s.rfind("+");

    //if (slash == 0 && bra < pls && s[bra + 1] != '+') {
      std::string name = s.substr(bra + 1, pls - bra - 1);
      std::string file = s.substr(0, bra);
      std::string offset = s.substr(pls + 1, ket - pls - 1);
      std::string address = s.substr(ebra + 1, eket - ebra - 1);

      int status = -1;
      std::string demangled;
      if (name.size() == 0) {
          demangled = "??";
      } else if (char *p = abi::__cxa_demangle(name.c_str(), NULL, NULL, &status); p) {
        demangled = std::string(p);
        free(p);
      } else {
        demangled = name;
      }
      //fmt::print(fg(fmt::color::red), "{}\n", s);
      fmt::print(fg(fmt::color::gray), "{}: ", address);
      fmt::print(fg(fmt::color::yellow), "{}", demangled);
      auto lineinfo = dbg::calc_addr_to_line(file, name, offset);
      if (lineinfo.size()) {
          fmt::print(fg(fmt::color::gray), " at ");
          fmt::print(fg(fmt::color::cyan), "{}", lineinfo);
      }
      fmt::print(fg(fmt::color::gray), " in ");
      fmt::print(fg(fmt::color::magenta), "{}", file);
      fmt::print(fg(fmt::color::gray), "\n");
    /*} else {
      fmt::print(fg(fmt::color::red), "{}\n", s);
    }*/
  }
  std::free(strings);
#endif
  fmt::print(fg(fmt::color::orange), "\nInternal error occurred.\n");
}
}
#else
namespace zeno {
void print_traceback(int skip) {
}
}
#endif
