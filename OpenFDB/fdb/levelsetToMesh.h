#include <fdb/schedule.h>
#include <set>
#include <vector>
#include <tuple>
#include <map>
#include <cstdio>
#include <cassert>

namespace fdb::levelsetToMesh {

inline static constexpr uint8_t NUM_VERTS_IN_TETRA = 4;
inline static constexpr uint8_t NUM_EDGES_IN_TETRA = 6;
inline static constexpr uint8_t NUM_VERTS_IN_CUBE = 8;
inline static constexpr uint8_t NUM_EDGES_IN_CUBE = 19;
inline static constexpr uint8_t NUM_TETRA_IN_CUBE = 6;

inline static const uint8_t TETRA_EDGE_TABLE[NUM_EDGES_IN_CUBE][2] =
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

inline static const uint8_t TETRA_VERTICES[NUM_TETRA_IN_CUBE][NUM_EDGES_IN_TETRA] =
{
  {0, 1, 5, 7},
  {0, 5, 4, 7},
  {0, 4, 6, 7},
  {0, 6, 2, 7},
  {0, 2, 3, 7},
  {0, 3, 1, 7}
};

inline static uint8_t TETRA_EDGES[NUM_EDGES_IN_TETRA][NUM_EDGES_IN_TETRA];

inline static uint8_t TETRA_VERTEX_TO_EDGE_MAP[NUM_VERTS_IN_CUBE][NUM_VERTS_IN_CUBE];
static bool construct_tetra_adjacency() {
  auto& tvem = TETRA_VERTEX_TO_EDGE_MAP;
  for (int i = 0; i < NUM_EDGES_IN_CUBE; ++i) {
    auto v0 = TETRA_EDGE_TABLE[i][0];
    auto v1 = TETRA_EDGE_TABLE[i][1];
    tvem[v0][v1] = i;
    tvem[v1][v0] = i;
  }

  for (int i = 0; i < NUM_EDGES_IN_TETRA; ++i) {
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
inline static bool cta = construct_tetra_adjacency();

inline static uint8_t TETRA_LOOKUP_PERM[16][4] = {
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

    {2, 3, 0, 1}, // 0b1100 ; two triangles, 2 and 3
    {1, 2, 3, 0}, // 0b1101 ; one flipped triangle, 1
    {0, 3, 2, 1}, // 0b1110 ; one flipped triangle, 0
    {0, 1, 2, 3}, // 0b1111 ; no triangles
};

template <class GridT>
struct MarchingTetra {
public:
MarchingTetra(GridT &sdf, float isovalue, float weldscale)
    : m_sdf(&sdf), m_isovalue(isovalue), m_weldscale(weldscale) {}
private:
GridT *m_sdf;
float m_isovalue;
float m_weldscale;

std::vector<std::pair<vec3i, vec3i>> m_tris;

inline void add_one_triangle_case(vec3i cube_idx, uint8_t i0, uint8_t i1, uint8_t i2, uint8_t i3) {
  auto e0 = TETRA_VERTEX_TO_EDGE_MAP[i0][i1];
  auto e1 = TETRA_VERTEX_TO_EDGE_MAP[i0][i2];
  auto e2 = TETRA_VERTEX_TO_EDGE_MAP[i0][i3];
  m_tris.emplace_back(cube_idx, vec3i(e0, e1, e2));
}

inline void add_two_triangles_case(vec3i cube_idx, uint8_t i0, uint8_t i1, uint8_t i2, uint8_t i3) {
  auto e0 = TETRA_VERTEX_TO_EDGE_MAP[i0][i2];
  auto e1 = TETRA_VERTEX_TO_EDGE_MAP[i0][i3];
  auto e2 = TETRA_VERTEX_TO_EDGE_MAP[i1][i2];
  auto e3 = TETRA_VERTEX_TO_EDGE_MAP[i1][i3];
  m_tris.emplace_back(cube_idx, vec3i(e0, e1, e2));
  m_tris.emplace_back(cube_idx, vec3i(e2, e1, e3));
}

inline float sample(int cx, int cy, int cz) {
    return m_sdf->get(vec3i(cx,cy,cz)) - m_isovalue;
}

void compute_cube(vec3i cube_index) {
  for (auto i = 0u; i < NUM_TETRA_IN_CUBE; ++i) {
    const auto& tv = TETRA_VERTICES[i];

    auto cx = cube_index[0];
    auto cy = cube_index[1];
    auto cz = cube_index[2];
    float vals[8] = {
        sample(cx, cy, cz),
        sample(cx+1, cy, cz),
        sample(cx, cy+1, cz),
        sample(cx+1, cy+1, cz),
        sample(cx, cy, cz+1),
        sample(cx+1, cy, cz+1),
        sample(cx, cy+1, cz+1),
        sample(cx+1, cy+1, cz+1),
    };

    // Create a lookup index based on the values
    uint8_t tri_lookup = 0;
    for (auto j = 0u; j < NUM_VERTS_IN_TETRA; ++j) {
      if (vals[tv[j]] > 0)
        tri_lookup |= 1 << j;
    }

    // Call the correct triangle addition case, modifying the lookups as necessary
    const auto& perm = TETRA_LOOKUP_PERM[tri_lookup];
    switch (tri_lookup) {
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
    }
  }
}

vec3i global_vertex_index(vec3i cube_index, uint8_t local_v) {
  size_t cx = cube_index[0];
  size_t cy = cube_index[1];
  size_t cz = cube_index[2];
    vec3i lut[8] = {
        vec3i(cx, cy, cz),
        vec3i(cx+1, cy, cz),
        vec3i(cx, cy+1, cz),
        vec3i(cx+1, cy+1, cz),
        vec3i(cx, cy, cz+1),
        vec3i(cx+1, cy, cz+1),
        vec3i(cx, cy+1, cz+1),
        vec3i(cx+1, cy+1, cz+1),
    };
  return lut[local_v];
}

vec3f get_edge_vertex_position(vec3i cube_index, int local_e) {
  auto& verts = TETRA_EDGE_TABLE[local_e];

  // global vertex indices
  auto vi0 = global_vertex_index(cube_index, verts[0]);
  auto vi1 = global_vertex_index(cube_index, verts[1]);

  float f0 = sample(vi0[0], vi0[1], vi0[2]), f1 = sample(vi1[0], vi1[1], vi1[2]);

  assert(f0 * f1 <= 0);

  // linear interpolate between the sampled points to find the zero-crossing
  float zero_t = f0 == f1 ? 0.5 : f0 / (f0 - f1);

  // interpolation between the vertex positions at the zero crossing
  return mix( vi0, vi1, zero_t );
};

std::vector<vec3f> m_vertices;
std::vector<vec3I> m_triangles;
std::map<std::tuple<int, int, int, int>, int> m_em;

protected:
void march_tetra() {
  for (int i = 0; i < m_tris.size(); i++) {
      for (int j = 0; j < 3; j++) {
          auto cube_idx = m_tris[i].first;
          auto local_e = m_tris[i].second[j];
          auto idx = std::make_tuple(cube_idx[0], cube_idx[1], cube_idx[2], local_e);
          if (m_em.find(idx) == m_em.end()) {
              m_em.emplace(idx, m_vertices.size());
              m_vertices.push_back(get_edge_vertex_position(cube_idx, local_e));
          }
        }
  }

  for (int i = 0; i < m_tris.size(); i++) {
      auto cube_idx = m_tris[i].first;
      auto ae = m_tris[i].second[0];
      auto a = std::make_tuple(cube_idx[0], cube_idx[1], cube_idx[2], ae);
      auto be = m_tris[i].second[1];
      auto b = std::make_tuple(cube_idx[0], cube_idx[1], cube_idx[2], be);
      auto ce = m_tris[i].second[2];
      auto c = std::make_tuple(cube_idx[0], cube_idx[1], cube_idx[2], ce);
      m_triangles.emplace_back(
              m_em.find(a)->second,
              m_em.find(b)->second,
              m_em.find(c)->second);
  }
}

void weld_close() {
    std::map<std::tuple<int, int, int>, std::vector<int>> rear;
    for (int i = 0; i < m_vertices.size(); i++) {
        auto pos = m_vertices[i];
        vec3i ipos(floor(pos * m_weldscale + 0.01f));
        rear[std::make_tuple(ipos[0], ipos[1], ipos[2])].push_back(i);
    }
    std::map<int, int> lut;
    std::vector<vec3f> new_verts;
    for (auto const &[ipos, inds]: rear) {
        vec3f cpos = m_vertices[inds[0]];
        int vertid = new_verts.size();
        lut.emplace(inds[0], vertid);
        for (int i = 1; i < inds.size(); i++) {
            cpos += m_vertices[inds[i]];
            lut.emplace(inds[i], vertid);
        }
        cpos /= inds.size();
        new_verts.emplace_back(cpos);
    }
    m_vertices = std::move(new_verts);
    std::set<std::tuple<int, int, int>> new_tris;
    for (auto const &inds: m_triangles) {
        new_tris.emplace(
                lut.find(inds[0])->second,
                lut.find(inds[1])->second,
                lut.find(inds[2])->second);
    }
    m_triangles.clear();
    for (auto const &[x, y, z]: new_tris) {
        if (x != y && y != z && z != x) {
            m_triangles.emplace_back(x, y, z);
        }
    }
}

void flip_edges() {
    std::set<std::tuple<int, int>> edges;
    for (auto const &ind: m_triangles) {
        auto x = ind[0], y = ind[1], z = ind[2];
        edges.emplace(x, y);
        edges.emplace(y, z);
        edges.emplace(z, x);
        edges.emplace(y, x);
        edges.emplace(z, y);
        edges.emplace(x, z);
    }

    std::vector<std::vector<int>> edgelut(m_vertices.size());
    for (auto const &[x, y]: edges) {
        edgelut[x].push_back(y);
    }

    std::map<std::tuple<int, int>, std::pair<int, int>> sevenedges;
    for (int i = 0; i < m_vertices.size(); i++) {
        if (edgelut[i].size() >= 7) {
            for (auto j: edgelut[i]) {
                if (edgelut[j].size() >= 7) {
                    sevenedges.emplace(std::make_tuple(i, j), std::make_pair(-1, -1));
                }
            }
        }
    }

    for (int i = 0; i < m_triangles.size(); i++) {
        auto &ind = m_triangles[i];
        std::tuple<int, int, int> enums[] = {
            {0, 1, 2}, {1, 2, 0}, {2, 0, 1},
            {1, 0, 2}, {2, 1, 0}, {0, 2, 1},
        };
        for (auto const &[a, b, c]: enums) {
            if (edgelut[ind[c]].size() > 6) {
                continue;
            }
            if (auto it = sevenedges.find({ind[a], ind[b]}); it != sevenedges.end()) {
                if (it->second.first != -1 && it->second.second != -1) {
                    if (edgelut[it->second.second].size() > 6) {
                        continue;
                    }
                    vec3I tmp_ind(ind[c], ind[a], it->second.second);
                    m_triangles[it->second.first] = vec3I(it->second.second, ind[b], ind[c]);
                    ind = tmp_ind;
                    it->second.first = -1;
                } else {
                    it->second = std::make_pair(i, ind[c]);
                }
            }
        }
    }
}

void smooth_mesh(int niters) {
    std::set<std::tuple<int, int>> edges;
    for (auto const &ind: m_triangles) {
        auto x = ind[0], y = ind[1], z = ind[2];
        edges.emplace(x, y);
        edges.emplace(y, z);
        edges.emplace(z, x);
        edges.emplace(y, x);
        edges.emplace(z, y);
        edges.emplace(x, z);
    }

    std::vector<std::vector<int>> edgelut(m_vertices.size());
    for (auto const &[x, y]: edges) {
        edgelut[x].push_back(y);
    }

    std::vector<vec3f> new_verts(m_vertices.size());
    for (int t = 0; t < niters; t++) {
        for (int i = 0; i < m_vertices.size(); i++) {
            vec3f avg = m_vertices[i] * edgelut[i].size();
            for (auto j: edgelut[i]) {
                avg += m_vertices[j];
            }
            avg /= 2 * edgelut[i].size();
            new_verts[i] = avg;
        }
        std::swap(new_verts, m_vertices);
    }
}

void compute_cubes() {
    //FILE *fp = fopen("a.txt", "w");
    m_sdf->foreach(Serial{}, [&] (auto idx, auto const &val) {
        compute_cube(idx);
    });
    //fclose(fp);
}

public:
void march() {
    printf("compute_cubes\n");
    compute_cubes();
    printf("march_tetra\n");
    march_tetra();
    printf("weld_close\n");
    weld_close();
    printf("flip_edges\n");
    flip_edges();
    printf("smooth_mesh\n");
    smooth_mesh(3);
    //printf("done\n");
}

inline auto const &triangles() const { return m_triangles; }
inline auto const &vertices() const { return m_vertices; }
inline auto &triangles() { return m_triangles; }
inline auto &vertices() { return m_vertices; }

};

template <class GridT>
auto marching_tetra
    ( GridT &grid
    , std::vector<vec3f> &vertices
    , std::vector<vec3I> &triangles
    , float isovalue = 0
    , float weldscale = 1
    ) {
    MarchingTetra mt(grid, isovalue, weldscale);
    mt.march();
    vertices = std::move(mt.vertices());
    triangles = std::move(mt.triangles());
}

}
