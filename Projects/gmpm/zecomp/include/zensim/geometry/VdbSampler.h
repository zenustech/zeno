#pragma once
#include <array>
#include <string>
#include <vector>

namespace zs {

  void initialize_openvdb();
  std::vector<std::array<float, 3>> sample_from_vdb_file(const std::string &filename, float dx,
                                                         float ppc);
  std::vector<std::array<float, 3>> sample_from_obj_file(const std::string &filename, float dx,
                                                         float ppc);

}  // namespace zs
