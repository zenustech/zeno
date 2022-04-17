#pragma once

#include <vector>
#include <functional>
#include "vec.h"

namespace fdb::marchingtetra {

using ImplicitEvalFunc = std::function<float(vec3i)>;

struct AABB {
    vec3f _min;
    vec3f _max;
    
    auto min() const { return _min; }
    auto max() const { return _max; }
};

class MarchingTetrahedra {
public:
  MarchingTetrahedra(ImplicitEvalFunc f, const AABB& box, size_t div);

  const auto& tris() const { return _tris; }
private:

  // Fill in the vector containing all of the grid-sampled points for the
  // implicit surface.
  void precompute_grid_vertices(ImplicitEvalFunc f);

  // Compute all of the triangles for a given cube.
  void compute_cube(size_t cx, size_t cy, size_t cz);

  void add_one_triangle_case(size_t cube_idx,
                             uint8_t i0, uint8_t i1, uint8_t i2, uint8_t i3);
  void add_two_triangles_case(size_t cube_idx,
                              uint8_t i0, uint8_t i1, uint8_t i2, uint8_t i3);
  void add_tri(size_t e0, size_t e1, size_t e2);

  // Build the list of vertices and a map from edge vertices to contiguous
  // vertices.
  void build_vertex_and_map(std::vector<vec3f> &) const;

  // Remap the triangle indices using the provided edge map, returning a new set
  // of triangles.
  std::vector<vec3I> remap_triangles() const;

  // Return a geometric vertex associated with a global edge index.
  vec3f get_edge_vertex_position(size_t e) const;

  vec3f get_vertex_position(size_t cube_index) const;

  // Return the global index of the edge, given the local vertices and the cube.
  size_t global_edge_index(size_t cube_index, uint8_t lv0, uint8_t lv1) const;

  // Return the global index of the vertex, given the cube index and local
  // vertex
  size_t global_vertex_index(size_t cube_index, uint8_t v) const;


  AABB _bbox;

  // points per size (one more than the number of cubes per side)
  size_t _pps;
  float _inv_div;

  // sampled at each grid point
  std::vector<float> _field_samples;

  // triangles in the resulting model
  std::vector<vec3I> _tris;

  size_t _lookup_offsets[8];

};

}
