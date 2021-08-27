#include <cstdio>
#include <cassert>
#include <fdb/schedule.h>
#include <fdb/VDBGrid.h>
#include <fdb/openvdb.h>
#include <set>
#include <vector>
#include <tuple>
#include <map>

using namespace fdb;

size_t g_nx = 64, g_ny = 64, g_nz = 64;

std::vector<vec3I> g_tris;

const uint8_t NUM_VERTS_IN_TETRA = 4;
const uint8_t NUM_EDGES_IN_TETRA = 6;
const uint8_t NUM_VERTS_IN_CUBE = 8;
const uint8_t NUM_EDGES_IN_CUBE = 19;
const uint8_t NUM_TETRA_IN_CUBE = 6;

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
bool construct_tetra_adjacency() {
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
static bool cta = construct_tetra_adjacency();

uint8_t TETRA_LOOKUP_PERM[16][4] = {
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
    {0, 1, 2, 3}, // 0b1111 ; no triangles
};

void add_tri(size_t e0, size_t e1, size_t e2) {
  g_tris.emplace_back(e0, e1, e2);
}

size_t global_edge_index(size_t cube_index, uint8_t lv0, uint8_t lv1) {
  return cube_index * NUM_EDGES_IN_CUBE + TETRA_VERTEX_TO_EDGE_MAP[lv0][lv1];
}

void add_one_triangle_case(size_t cube_idx, uint8_t i0, uint8_t i1, uint8_t i2, uint8_t i3) {

  auto e0 = global_edge_index(cube_idx, i0, i1);
  auto e1 = global_edge_index(cube_idx, i0, i2);
  auto e2 = global_edge_index(cube_idx, i0, i3);
  add_tri(e0, e1, e2);
}

void add_two_triangles_case(size_t cube_idx, uint8_t i0, uint8_t i1, uint8_t i2, uint8_t i3) {
  auto e0 = global_edge_index(cube_idx, i0, i2);
  auto e1 = global_edge_index(cube_idx, i0, i3);
  auto e2 = global_edge_index(cube_idx, i1, i2);
  auto e3 = global_edge_index(cube_idx, i1, i3);
  add_tri(e0, e1, e2);
  add_tri(e2, e1, e3);
}

vdbgrid::VDBGrid<float> g_sdf;

float sample(size_t cx, size_t cy, size_t cz) {
    return g_sdf.at(vec3i(cx,cy,cz));
}

void compute_cube(size_t cx, size_t cy, size_t cz) {
  for (auto i = 0u; i < NUM_TETRA_IN_CUBE; ++i) {
    const auto& tv = TETRA_VERTICES[i];

    size_t cube_index = cz * g_ny * g_nx + cy * g_nx + cx;

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
    for (auto j = 0u; j < NUM_VERTS_IN_TETRA; ++j)
      tri_lookup |= (vals[tv[j]] > 0 ? 1 : 0) << j;

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

vec3i global_vertex_index(size_t cube_index, uint8_t local_v) {
  size_t cx = cube_index % g_nx;
  cube_index /= g_nx;
  size_t cy = cube_index % g_ny;
  cube_index /= g_ny;
  size_t cz = cube_index;
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

vec3f get_edge_vertex_position(size_t e) {
  size_t cube_index = e / NUM_EDGES_IN_CUBE;
  uint8_t local_e = e % NUM_EDGES_IN_CUBE;

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

std::vector<vec3f> g_vertices;
std::vector<vec3I> g_triangles;
std::map<int, int> g_em;

void marching_tetra() {
  for (size_t cz = 0; cz < g_nz; ++cz)
    for (size_t cy = 0; cy < g_ny; ++cy)
      for (size_t cx = 0; cx < g_nx; ++cx)
        compute_cube(cx, cy, cz);

  for (int i = 0; i < g_tris.size(); i++) {
      for (int j = 0; j < 3; j++) {
          auto idx = g_tris[i][j];
          if (g_em.find(idx) == g_em.end()) {
              g_em.emplace(idx, g_vertices.size());
              g_vertices.push_back(get_edge_vertex_position(idx));
          }
        }
  }

  for (int i = 0; i < g_tris.size(); i++) {
      g_triangles.emplace_back(
              g_em.find(g_tris[i][0])->second,
              g_em.find(g_tris[i][1])->second,
              g_em.find(g_tris[i][2])->second);
  }
}

void weld_close() {
    std::map<std::tuple<int, int, int>, std::vector<int>> rear;
    for (int i = 0; i < g_vertices.size(); i++) {
        auto pos = g_vertices[i];
        vec3i ipos(pos);
        rear[std::make_tuple(ipos[0], ipos[1], ipos[2])].push_back(i);
    }
    std::map<int, int> lut;
    std::vector<vec3f> new_verts;
    for (auto const &[ipos, inds]: rear) {
        vec3f cpos = g_vertices[inds[0]];
        int vertid = new_verts.size();
        lut.emplace(inds[0], vertid);
        for (int i = 1; i < inds.size(); i++) {
            cpos += g_vertices[inds[i]];
            lut.emplace(inds[i], vertid);
        }
        cpos /= inds.size();
        new_verts.emplace_back(cpos);
    }
    g_vertices = std::move(new_verts);
    std::set<std::tuple<int, int, int>> new_tris;
    for (auto const &inds: g_triangles) {
        new_tris.emplace(
                lut.find(inds[0])->second,
                lut.find(inds[1])->second,
                lut.find(inds[2])->second);
    }
    g_triangles.clear();
    for (auto const &[x, y, z]: new_tris) {
        if (x != y && y != z && z != x) {
            g_triangles.emplace_back(x, y, z);
        }
    }
}

std::tuple<int, int, int> sort_three(vec3i ind) {
    int i1 = 0, i2 = 1, i3 = 2;
    if (ind[i1] > ind[i2]) {
        std::swap(i1, i2);
    }
    if (ind[i1] > ind[i3]) {
        std::swap(i1, i3);
    }
    if (ind[i2] > ind[i3]) {
        std::swap(i2, i3);
    }
    return {ind[i1], ind[i2], ind[i3]};
}

void flip_edges() {
    std::set<std::tuple<int, int>> edges;
    for (auto const &ind: g_triangles) {
        auto x = ind[0], y = ind[1], z = ind[2];
        edges.emplace(x, y);
        edges.emplace(y, z);
        edges.emplace(z, x);
        edges.emplace(y, x);
        edges.emplace(z, y);
        edges.emplace(x, z);
    }

    std::vector<std::vector<int>> edgelut(g_vertices.size());
    for (auto const &[x, y]: edges) {
        edgelut[x].push_back(y);
        //edgelut[y].push_back(x);
    }

    std::map<std::tuple<int, int>, std::pair<int, int>> sevenedges;
    for (int i = 0; i < g_vertices.size(); i++) {
        if (edgelut[i].size() >= 7) {
            for (auto j: edgelut[i]) {
                if (edgelut[j].size() >= 7) {
                    sevenedges.emplace(std::make_tuple(i, j), std::make_pair(-1, -1));
                }
            }
        }
    }

    for (int i = 0; i < g_triangles.size(); i++) {
        auto &ind = g_triangles[i];
        std::tuple<int, int, int> enums[] = {
            {0, 1, 2}, {1, 2, 0}, {2, 0, 1},
            {1, 0, 2}, {2, 1, 0}, {0, 2, 1},
        };
        for (auto const &[a, b, c]: enums) {
            if (auto it = sevenedges.find({ind[a], ind[b]}); it != sevenedges.end()) {
                if (it->second.first != -1) {
                    printf("second %d %d %d\n", ind[a], ind[b], ind[c]);
                    vec3I tmp_ind(ind[c], ind[a], it->second.second);
                    g_triangles[it->second.first] = vec3I(it->second.second, ind[b], ind[c]);
                    ind = tmp_ind;
                    it->second.first = -1;
                } else {
                    printf("first %d %d %d\n", ind[a], ind[b], ind[c]);
                    it->second = std::make_pair(i, ind[c]);
                }
            }
        }
    }
}

int main() {
    ndrange_for(Serial{}, vec3i(0), vec3i(65), [&] (auto idx) {
        float value = max(-4.0f, length(tofloat(idx)) - 16.9f);
        g_sdf.set(idx, value);
    });

    marching_tetra();
    weld_close();
    flip_edges();

    FILE *fp = fopen("/tmp/a.obj", "w");
    for (auto f: g_triangles) { f += 1;
        fprintf(fp, "f %d %d %d\n", f[0], f[1], f[2]);
    }
    for (auto v: g_vertices) {
        fprintf(fp, "v %f %f %f\n", v[0], v[1], v[2]);
    }
    fclose(fp);

    /*write_dense_vdb("/tmp/a.vdb", [&] (auto idx) {
        return g_sdf.get(idx);
    }, vec3i(0), vec3i(64));*/

    return 0;
}
