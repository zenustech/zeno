/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file (MIT LICENSE).
*******************************************************************************/
#pragma once
#include <string_view>

namespace zs {

  class DynamicLoader {
  private:
    void load_dll(std::string_view dll_path);
    void close_dll();

  public:
    DynamicLoader(std::string_view dll_path);
    void *load_function(std::string_view func_name);

    template <typename T> void load_function(std::string_view func_name, T &f) {
      f = (T)load_function(func_name);
    }

    bool loaded() const noexcept;

    ~DynamicLoader();

  private:
    void *dll{nullptr};
  };

}  // namespace zs
