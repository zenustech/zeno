#pragma once
#include <cassert>
#include <array>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace zs {
  template <typename T, int dim, typename Tn = int, int dimE = dim + 1> struct Mesh {
    using Node = std::array<T, dim>;
    using Elem = std::array<Tn, dimE>;
    std::vector<Node> nodes;
    std::vector<Elem> elems;
  };
}  // namespace zs


namespace zs {
  template <typename T, typename Tn>
  bool read_tet_mesh_vtk(const std::string &file, Mesh<T, 3, Tn, 4> &mesh) {
    // TetMesh
    std::ifstream in(file);
    if (!in || file.empty()) {
      printf("%s not found!\n", file.c_str());
      return false;
    }

    using Node = typename Mesh<T, 3, Tn, 4>::Node;
    using Elem = typename Mesh<T, 3, Tn, 4>::Elem;
    auto &X = mesh.nodes;
    auto &indices = mesh.elems;

    Tn initial_X_size = X.size();
    Tn initial_indices_size = indices.size();

    std::string line;
    Node position;

    bool reading_points = false;
    bool reading_tets = false;
    std::size_t n_points = 0;
    std::size_t n_tets = 0;

    while (std::getline(in, line)) {
      std::stringstream ss(line);
      if (line.size() == (std::size_t)(0)) {
      } else if (line.substr(0, 6) == "POINTS") {
        reading_points = true;
        reading_tets = false;
        ss.ignore(128, ' ');  // ignore "POINTS"
        ss >> n_points;
      } else if (line.substr(0, 5) == "CELLS") {
        reading_points = false;
        reading_tets = true;
        ss.ignore(128, ' ');  // ignore "CELLS"
        ss >> n_tets;
      } else if (line.substr(0, 10) == "CELL_TYPES") {
        reading_points = false;
        reading_tets = false;
      } else if (reading_points) {
        for (int i = 0; i < 3; i++) ss >> position[i];
        X.emplace_back(position);
      } else if (reading_tets) {
        ss.ignore(128, ' ');  // ignore "4"
        Elem tet;
        for (int i = 0; i < 4; i++) {
          ss >> tet[i];
          tet[i] += initial_X_size;
        }
        indices.emplace_back(tet);
      }
    }
    in.close();

    assert((n_points == X.size() - initial_X_size) && "vtk read X count doesn't match.");
    assert((n_tets == indices.size() - initial_indices_size)
           && "vtk read element count doesn't match.");
    printf("positions, tetrahedra [%d, %d] -> [%d, %d]\n", initial_X_size, initial_indices_size,
           (int)X.size(), (int)indices.size());
    return true;
  }
}  // namespace zs
