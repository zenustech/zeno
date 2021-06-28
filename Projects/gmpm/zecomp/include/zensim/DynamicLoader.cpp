#include "DynamicLoader.h"

#include "Logger.hpp"
#include "Platform.hpp"
#if defined(ZS_PLATFORM_WINDOWS)
#  include <windows.h>
// Never directly include <windows.h>. That will bring you evil max/min macros.
#  if defined(min)
#    undef min
#  endif
#  if defined(max)
#    undef max
#  endif

#else
#  include <dlfcn.h>
#endif

namespace zs {
  DynamicLoader::DynamicLoader(std::string_view dll_path) { load_dll(dll_path); }

  void DynamicLoader::load_dll(std::string_view dll_path) {
#ifdef WIN32
    dll = (HMODULE)LoadLibraryA(dll_path.data());
#else
    dll = dlopen(dll_path.data(), RTLD_LAZY);
#endif
  }

  void *DynamicLoader::load_function(std::string_view func_name) {
    ZS_ERROR_IF(!loaded(), "DLL not opened");
#if defined(ZS_PLATFORM_WINDOWS)
    auto func = (void *)GetProcAddress((HMODULE)dll, func_name.data());
#else
    auto func = dlsym(dll, func_name.data());
    const char *dlsym_error = dlerror();
    ZS_ERROR_IF(dlsym_error, "Cannot load function: {}", dlsym_error);
#endif
    ZS_ERROR_IF(!func, "Function {} not found", func_name);
    return func;
  }

  void DynamicLoader::close_dll() {
    ZS_ERROR_IF(!loaded(), "DLL not opened");
#if defined(ZS_PLATFORM_WINDOWS)
    FreeLibrary((HMODULE)dll);
#else
    dlclose(dll);
#endif
    dll = nullptr;
  }

  DynamicLoader::~DynamicLoader() {
    if (loaded()) close_dll();
  }

  bool DynamicLoader::loaded() const noexcept { return dll != nullptr; }

}  // namespace zs
