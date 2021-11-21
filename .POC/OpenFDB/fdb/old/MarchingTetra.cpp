#include "MarchingTetra.h"
#include <cassert>

namespace fdb::marchingtetra {

using std::make_pair;

const uint8_t NUM_VERTS_IN_TETRA = 4;
const uint8_t NUM_EDGES_IN_TETRA = 6;
const uint8_t NUM_VERTS_IN_CUBE = 8;
const uint8_t NUM_EDGES_IN_CUBE = 19;
const uint8_t NUM_TETRA_IN_CUBE = 6;

template <class A, class B, class T>
auto lerp(A a, B b, T t) {
    return a + (b - a) * t;
}

MarchingTetrahedra::MarchingTetrahedra(ImplicitEvalFunc f, const AABB& box, size_t div)
  : _bbox(box), _pps(div+1), _inv_div(1.0f/div),  _field_samples(_pps*_pps*_pps),
    _lookup_offsets{0, 1, _pps, _pps + 1,
                     _pps*_pps, _pps*_pps + 1, _pps*_pps + _pps, _pps*_pps + _pps + 1}
{
  precompute_grid_vertices(f);

  for (size_t cz = 0; cz < div; ++cz)
    for (size_t cy = 0; cy < div; ++cy)
      for (size_t cx = 0; cx < div; ++cx)
        compute_cube(cx, cy, cz);
}

const uint8_t TETRA_EDGE_TABLE[NUM_EDGES_IN_CUBE][2] =
{{0, 1}, // around z = 0
 {1, 3},
 {2, 3},
 {0, 2},

 {4, 5}, // around z = 1
 {5, 7},
 {6, 7},
 {4, 6},

 {0, 4}, // normal to x-y
 {1, 5},
 {2, 6},
 {3, 7},

 {0, 3}, // face diagonal, 0
 {0, 6},
 {0, 5},

 {4, 7}, // face diagonal, 1
 {1, 7},
 {2, 7},

 {0, 7} // cross diagonal
};

const uint8_t TETRA_VERTICES[NUM_TETRA_IN_CUBE][NUM_EDGES_IN_TETRA] =
{
  {0, 1, 5, 7},
  {0, 5, 4, 7},
  {0, 4, 6, 7},
  {0, 6, 2, 7},
  {0, 2, 3, 7},
  {0, 3, 1, 7}
};

static uint8_t TETRA_EDGES[NUM_EDGES_IN_TETRA][NUM_EDGES_IN_TETRA];

static uint8_t TETRA_VERTEX_TO_EDGE_MAP[NUM_VERTS_IN_CUBE][NUM_VERTS_IN_CUBE];

bool construct_tetra_adjacency()
{
  auto& tvem = TETRA_VERTEX_TO_EDGE_MAP;
  for (int i = 0; i < NUM_EDGES_IN_CUBE; ++i)
  {
    auto v0 = TETRA_EDGE_TABLE[i][0];
    auto v1 = TETRA_EDGE_TABLE[i][1];
    tvem[v0][v1] = i;
    tvem[v1][v0] = i;
  }

  for (int i = 0; i < NUM_EDGES_IN_TETRA; ++i)
  {
    auto& te = TETRA_EDGES[i];
    const auto& tv = TETRA_VERTICES[i];
    te[0] = tvem[tv[0]][tv[1]];
    te[1] = tvem[tv[0]][tv[2]];
    te[2] = tvem[tv[0]][tv[3]];
    te[3] = tvem[tv[1]][tv[2]];
    te[4] = tvem[tv[1]][tv[3]];
    te[5] = tvem[tv[2]][tv[3]];
  }

  return true;
}

static bool cta = construct_tetra_adjacency();

void MarchingTetrahedra::precompute_grid_vertices(ImplicitEvalFunc f)
{
  // Fill the field samples vector
  for (size_t i = 0, iz = 0; iz < _pps; ++iz)
  {
    float z = lerp(_bbox.min()[2], _bbox.max()[2], iz * _inv_div);
    for (size_t iy = 0; iy < _pps; ++iy)
    {
      float y = lerp(_bbox.min()[1], _bbox.max()[1], iy * _inv_div);
      for (size_t ix = 0; ix < _pps; ++ix, ++i)
      {
        float x = lerp(_bbox.min()[0], _bbox.max()[0], ix * _inv_div);
        vec3f v{x, y, z};
        _field_samples[i] = f(v);
      }
    }
  }
}

uint8_t TETRA_LOOKUP_PERM[16][4] =
{
  {0, 1, 2, 3}, // 0b0000 ; no triangles
  {0, 1, 2, 3}, // 0b0001 ; one triangle, vertex 0
  {1, 2, 0, 3}, // 0b0010 ; one triangle, vertex 1
  {0, 1, 2, 3}, // 0b0011 ; two triangles, 0 and 1

  {2, 3, 0, 1}, // 0b0100 ; one triangle, vertex 2
  {0, 2, 3, 1}, // 0b0101 ; two triangles, 0 and 2
  {1, 2, 0, 3}, // 0b0110 ; two triangles, 1 and 2
  {3, 0, 1, 2}, // 0b0111 ; one flipped triangle 3

  {3, 0, 2, 1}, // 0b1000 ; one triangle, 3
  {3, 0, 2, 1}, // 0b1001 ; two triangles, 0 and 3
  {1, 3, 2, 0}, // 0b1010 ; two triangles, 1 and 3
  {2, 3, 1, 0}, // 0b1011 ; one flipped triangle, 2

  {2 ,3, 0, 1}, // 0b1100 ; two triangles, 2 and 3
  {1, 2, 3, 0}, // 0b1101 ; one flipped triangle, 1
  {0, 3, 2, 1}, // 0b1110 ; one flipped triangle, 0
  {0, 1, 2, 3}  // 0b1111 ; no triangles
};

void MarchingTetrahedra::compute_cube(size_t cx, size_t cy, size_t cz)
{
  float vals[8];
  size_t cube_index = cz * (_pps - 1) * (_pps - 1) + cy * (_pps - 1) + cx;
  size_t base_offset = cz * _pps * _pps + cy * _pps + cx;

  // Retrieve the values at the eight vertices
  for (int i = 0; i < 8; ++i)
    vals[i] = _field_samples[base_offset + _lookup_offsets[i]];

  for (auto i = 0u; i < NUM_TETRA_IN_CUBE; ++i)
  {
    const auto& tv = TETRA_VERTICES[i];

    // Create a lookup index based on the values
    uint8_t tri_lookup = 0;
    for (auto j = 0u; j < NUM_VERTS_IN_TETRA; ++j)
      tri_lookup |= (vals[tv[j]] > 0 ? 1 : 0) << j;

    // Call the correct triangle addition case, modifying the lookups as necessary
    const auto& perm = TETRA_LOOKUP_PERM[tri_lookup];
    switch (tri_lookup)
    {
    case 0b0000:
    case 0b1111:
      break;
    case 0b0001:
    case 0b0010:
    case 0b0100:
    case 0b1000:
    case 0b1110:
    case 0b1101:
    case 0b1011:
    case 0b0111:
      add_one_triangle_case(cube_index,
                            tv[perm[0]], tv[perm[1]],
                            tv[perm[2]], tv[perm[3]]);
      break;
    case 0b0011:
    case 0b0101:
    case 0b0110:
    case 0b1001:
    case 0b1010:
    case 0b1100:
      add_two_triangles_case(cube_index,
                             tv[perm[0]], tv[perm[1]],
                             tv[perm[2]], tv[perm[3]]);
      break;
    default:
      assert("Invalid tri_lookup" && false);
    }
  }
}

size_t MarchingTetrahedra::global_edge_index(size_t cube_index, uint8_t lv0, uint8_t lv1) const
{
  return cube_index * NUM_EDGES_IN_CUBE + TETRA_VERTEX_TO_EDGE_MAP[lv0][lv1];
}

void MarchingTetrahedra::add_one_triangle_case(size_t cube_idx,
                                               uint8_t i0, uint8_t i1, uint8_t i2, uint8_t i3)
{

  auto e0 = global_edge_index(cube_idx, i0, i1);
  auto e1 = global_edge_index(cube_idx, i0, i2);
  auto e2 = global_edge_index(cube_idx, i0, i3);
  add_tri(e0, e1, e2);
}

void MarchingTetrahedra::add_two_triangles_case(size_t cube_idx,
                                                uint8_t i0, uint8_t i1, uint8_t i2, uint8_t i3)
{
  auto e0 = global_edge_index(cube_idx, i0, i2);
  auto e1 = global_edge_index(cube_idx, i0, i3);
  auto e2 = global_edge_index(cube_idx, i1, i2);
  auto e3 = global_edge_index(cube_idx, i1, i3);

  add_tri(e0, e1, e2);
  add_tri(e2, e1, e3);
}

void MarchingTetrahedra::add_tri(size_t e0, size_t e1, size_t e2)
{
  _tris.emplace_back(e0, e1, e2);
}

vec3f MarchingTetrahedra::get_vertex_position(size_t vi) const
{
  assert(vi < _field_samples.size());
  size_t cx = vi % _pps;
  vi /= _pps;
  size_t cy = vi % _pps;
  vi /= _pps;
  size_t cz = vi; // % _pps;

  auto v = lerp(_bbox.min(), _bbox.max(), vec3f(cx, cy, cz) * _inv_div);
  return v;
}

size_t MarchingTetrahedra::global_vertex_index(size_t cube_index, uint8_t local_v) const
{
  const size_t div = _pps - 1;
  size_t cx = cube_index % div;
  cube_index /= div;
  size_t cy = cube_index % div;
  cube_index /= div;
  size_t cz = cube_index;

  return ((cz * _pps) + cy) * _pps + cx + _lookup_offsets[local_v];
}

vec3f MarchingTetrahedra::get_edge_vertex_position(size_t e) const
{
  size_t cube_index = e / NUM_EDGES_IN_CUBE;
  uint8_t local_e = e % NUM_EDGES_IN_CUBE;

  auto& verts = TETRA_EDGE_TABLE[local_e];

  // global vertex indices
  size_t vi0 = global_vertex_index(cube_index, verts[0]);
  size_t vi1 = global_vertex_index(cube_index, verts[1]);

  assert(vi0 < _field_samples.size());
  assert(vi1 < _field_samples.size());

  float f0 = _field_samples[vi0], f1 = _field_samples[vi1];

  assert(f0 * f1 <= 0);

  // linear interpolate between the sampled points to find the zero-crossing
  float zero_t = f0 == f1 ? 0.5f : f0 / (f0 - f1);

  // interpolation between the vertex positions at the zero crossing
  return lerp( get_vertex_position(vi0), get_vertex_position(vi1), zero_t );
};

}
