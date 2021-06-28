#pragma once
#include <string>

#include "../Platform.hpp"

namespace zs {

  std::string get_cu_error_message(uint32_t err);
  std::string get_cuda_error_message(uint32_t err);

  template <typename... Args> class CudaDriverApi {
  public:
    CudaDriverApi() { function = nullptr; }

    void set(void *func_ptr) { function = (func_type *)func_ptr; }

    uint32_t call(Args... args) {
      assert(function != nullptr);
      return (uint32_t)function(args...);
    }

    void set_names(const std::string &name, const std::string &symbol_name) {
      this->name = name;
      this->symbol_name = symbol_name;
    }

#if 0
    std::string get_error_message(uint32_t err) {
      return get_cu_error_message(err) + fmt::format(" while calling {} ({})", name, symbol_name);
    }
#endif

    uint32_t call_with_warning(Args... args) {
      auto err = call(args...);
      // ZS_WARN_IF(err, "{}", get_error_message(err));
      return err;
    }

    // Note: CUDA driver API passes everything as value
    void operator()(Args... args) {
      auto err = call(args...);
      // ZS_ERROR_IF(err, get_error_message(err));
    }

  private:
    using func_type = uint32_t(Args...);

    func_type *function{nullptr};
    std::string name, symbol_name;
  };

  template <typename... Args> class CudaRuntimeApi {
  public:
    CudaRuntimeApi() { function = nullptr; }

    void set(void *func_ptr) { function = (func_type *)func_ptr; }

    uint32_t call(Args... args) {
      assert(function != nullptr);
      return (uint32_t)function(args...);
    }

    void set_names(const std::string &name, const std::string &symbol_name) {
      this->name = name;
      this->symbol_name = symbol_name;
    }

#if 0
    std::string get_error_message(uint32_t err) {
      return get_cuda_error_message(err) + fmt::format(" while calling {} ({})", name, symbol_name);
    }
#endif

    uint32_t call_with_warning(Args... args) {
      auto err = call(args...);
      // ZS_WARN_IF(err, "{}", get_error_message(err));
      return err;
    }

    void operator()(Args... args) {
      auto err = call(args...);
      // ZS_ERROR_IF(err, get_error_message(err));
    }

  private:
    using func_type = uint32_t(Args...);

    func_type *function{nullptr};
    std::string name, symbol_name;
  };

}  // namespace zs
