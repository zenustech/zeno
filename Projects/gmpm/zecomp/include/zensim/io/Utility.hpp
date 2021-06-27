#ifndef __IO_UTILITY_HPP__
#define __IO_UTILITY_HPP__
#include <string>

namespace zs {

  std::string file_get_content(std::string const &path);
  void *load_raw_file(char const *filename, size_t size);

}  // namespace zs

#endif
