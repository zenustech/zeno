#ifndef SPARSE_BUFFER_H
#define SPARSE_BUFFER_H

#include "tbb/tbb.h"
#include "tile.h"
#include "util.h"
#include "vec.h"
#include <assert.h>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <algorithm>
#include <execution>
#include <numeric>

#include <fmt/color.h>
#include <fmt/core.h>
#include "openvdb/tools/GridOperators.h"
#include "openvdb/tools/Interpolation.h"
#include "openvdb/tools/LevelSetFilter.h"
#include "openvdb/tools/LevelSetPlatonic.h"
#include "openvdb/tools/LevelSetSphere.h"
#include "openvdb/tools/MeshToVolume.h"
#include "openvdb/tools/ParticlesToLevelSet.h"
#include "openvdb/tools/VolumeToMesh.h"
#include <openvdb/Exceptions.h>
#include <openvdb/Types.h>
#include <openvdb/math/Math.h>
#include <openvdb/openvdb.h>
#include <openvdb/tree/LeafNode.h>
#include <openvdb/util/Util.h>
#include "flipParticle.h"
#include "volumeMeshTools.h"
struct iSpaceHasher {
  uint64 operator()(const FLUID::Vec3i &k) const {
    return ((k[0] + 1024) % 1024) +
            1024 * (((k[1] + 1024) % 1024) + 1024 * ((k[2] + 1024) % 1024));
  }
};
struct iequal_to {
  bool operator()(const FLUID::Vec3i &a, const FLUID::Vec3i &b) const {
    return ((a[0] == b[0]) && (a[1] == b[1]) && (a[2] == b[2]));
  }
};
using namespace std;

template <int N> struct sparse_fluid_3D {
    using GridT =
    typename openvdb::Grid<typename openvdb::tree::Tree4<float, 5, 4, 3>::Type>;
    using TreeT = typename GridT::TreeType;
  float h, bulk_size;
  int tile_n;
  uint n_bulks=0, n_perbulk;
  vector<fluid_Tile<N>> fluid_bulk;
  unordered_map<FLUID::Vec3i, uint64, iSpaceHasher, iequal_to>
      index_mapping; // maps an bulk's physical coord to bulk array
  // indexMapper index_mapping;
  vector<FLUID::Vec3i> loop_order;
  uint ni, nj, nk;
  FLUID::Vec3f bmin, bmax;
  std::string domainName;

  void extend_fluidbulks() {
    vector<fluid_Tile<N>> new_fluid_bulk;
    new_fluid_bulk.resize(0);
    for (int i = 0; i < fluid_bulk.size(); i++) {
      new_fluid_bulk.push_back(
          fluid_Tile<N>(fluid_bulk[i].tile_corner, fluid_bulk[i].bulk_index));
    }
    for (int i = 0; i < fluid_bulk.size(); i++) {
      FLUID::Vec3i bulk_ijk = fluid_bulk[i].tile_corner / N;
      for (int kk = bulk_ijk[2] - 1; kk <= bulk_ijk[2] + 1; kk++)
        for (int jj = bulk_ijk[1] - 1; jj <= bulk_ijk[1] + 1; jj++)
          for (int ii = bulk_ijk[0] - 1; ii <= bulk_ijk[0] + 1; ii++) {
            if (ii >= 0 && ii < ni && jj >= 0 && jj < nj && kk >= 0 &&
                kk < nk) {

              // uint64 idx = kk*nj*ni + jj*ni + ii;
              FLUID::Vec3i alc_ijk(ii, jj, kk);
              if (index_mapping.find(alc_ijk) == index_mapping.end()) {
                index_mapping[alc_ijk] = new_fluid_bulk.size();
                new_fluid_bulk.push_back(
                    fluid_Tile<N>(FLUID::Vec3i(ii * N, jj * N, kk * N),
                                  new_fluid_bulk.size()));
              }
            }
          }
    }
    fluid_bulk.resize(0);
    for (int i = 0; i < new_fluid_bulk.size(); i++) {
      fluid_bulk.push_back(fluid_Tile<N>(new_fluid_bulk[i].tile_corner,
                                         new_fluid_bulk[i].bulk_index));
    }
  }
  void initialize_bulks(typename GridT::Ptr grid, double cell_size) {
      fmt::print("0 begin init bulks\n");
      // static int prevSize = 100000;
      // fluid_bulk.reserve(prevSize * 2);
      // fluid_bulk.reserve(particles.size());
      // fluid_bulk.resize(particles.size());

      fmt::print("1 finish reserve\n");
      // fluid_bulk.resize(0);
      index_mapping.clear();
      h = cell_size;
      uint num_particles = grid->activeVoxelCount();
      openvdb::CoordBBox box = grid->evalActiveVoxelBoundingBox();
      auto world_min = grid->indexToWorld(box.min());
      auto world_max = grid->indexToWorld(box.max());
      bmin = FLUID::Vec3f(world_min[0], world_min[1], world_min[2]);
      bmax = FLUID::Vec3f(world_max[0], world_max[1], world_max[2]);

      fmt::print("2 finish boundary compute\n");


      bulk_size = (float)N * h;
      ni = ceil((bmax[0] - bmin[0]) / bulk_size);
      nj = ceil((bmax[1] - bmin[1]) / bulk_size);
      nk = ceil((bmax[2] - bmin[2]) / bulk_size);
      std::vector<char> buffer;
      std::vector<int> indexBuffer(ni * nj * nk);
      buffer.resize(ni * nj * nk);
      buffer.assign(ni * nj * nk, 0);
      fmt::print("3 finish hash buffer init\n");
      // abuffer.assign(ni * nj * nk, 0);
      // tbb::parallel_for((uint)0, (uint)ni * nj * nk, (uint)1,
      //                  [&abuffer](uint i) { abuffer[i] = 0; });
      tbb::parallel_for((uint)0, (uint)ni * nj * nk, (uint)1, [&](uint index) {
          int i=index%ni;
          int j=(index/ni)%nj;
          int k=index/(ni*nj);
          for(int ii=0;ii<8;ii++)for(int jj=0;jj<8;jj++)for(int kk=0;kk<8;kk++)
          {
              FLUID::Vec3f pos = bmin + FLUID::Vec3f(i*8+ii, j*8+jj, k*8+kk)*cell_size;
              openvdb::tools::GridSampler<TreeT, openvdb::tools::BoxSampler> interpolator(
                      grid->constTree(), grid->transform());
              openvdb::math::Vec3<float> P(pos[0], pos[1], pos[2]);
              if(interpolator.wsSample(P)<0)
              {
                  buffer[index] = 1;
              }
          }
      });
      fmt::print("4 finish hashing\n");
      std::exclusive_scan(std::execution::par, buffer.begin(), buffer.end(),
                          indexBuffer.begin(), 0);
      fmt::print("5 finish scanning\n");

      auto cnt = indexBuffer.back() + buffer.back();
      fluid_bulk.resize(cnt);
      tbb::parallel_for(
              (uint)0, (uint)buffer.size(), (uint)1, [&](uint buffer_id) {
                  if (buffer[buffer_id] == 1) {
                      int ii = buffer_id % ni, jj = (buffer_id % (ni * nj)) / ni,
                              kk = buffer_id / (ni * nj);
                      int id = indexBuffer[buffer_id];
                      fluid_bulk[id] =
                              fluid_Tile<N>(FLUID::Vec3i(N * ii, N * jj, N * kk), id);
                  }
              });
      fmt::print("6 finish bulk list build\n");

      for (int buffer_id = 0; buffer_id < fluid_bulk.size(); buffer_id++) {
          FLUID::Vec3i corner = fluid_bulk[buffer_id].tile_corner;
          int ii = corner[0] / N;
          int jj = corner[1] / N;
          int kk = corner[2] / N;
          FLUID::Vec3i bulk_ijk = FLUID::Vec3i(ii, jj, kk);
          index_mapping[bulk_ijk] = buffer_id;
      }
      fmt::print("7 finish bulk mapping build\n");
      fmt::print("identify boundary bulks\n");
      n_bulks = fluid_bulk.size();
      cout << "num of bulks:" << fluid_bulk.size() << endl;
      cout << "bmin:" << bmin[0] << " " << bmin[1] << " " << bmin[2] << endl;
      cout << "bmax:" << bmax[0] << " " << bmax[1] << " " << bmax[2] << endl;
      cout << "dimension:" << ni << " " << nj << " " << nk << endl;
  }
  void initialize_bulks(vector<FLIP_particle> &particles, double cell_size) {

    fmt::print("0 begin init bulks\n");
    // static int prevSize = 100000;
    // fluid_bulk.reserve(prevSize * 2);
    // fluid_bulk.reserve(particles.size());
    // fluid_bulk.resize(particles.size());

    fmt::print("1 finish reserve\n");
    // fluid_bulk.resize(0);
    index_mapping.clear();
    h = cell_size;
    uint num_particles = particles.size();
    bmin = particles[0].pos;
    bmax = particles[0].pos;
    for (uint i = 1; i < num_particles; i++) {
      bmin = min_union(bmin, particles[i].pos);
      bmax = max_union(bmax, particles[i].pos);
    }
    bmin -= FLUID::Vec3f(24 * h, 24 * h, 24 * h);
    bmax += FLUID::Vec3f(24 * h, 24 * h, 24 * h);
    bulk_size = (float)N * h;
    ni = ceil((bmax[0] - bmin[0]) / bulk_size);
    nj = ceil((bmax[1] - bmin[1]) / bulk_size);
    nk = ceil((bmax[2] - bmin[2]) / bulk_size);
    /*FLUID::Vec3f bmin = h*FLUID::Vec3f(floor(bmin[0]/h),
    floor(bmin[1]/h),
    floor(bmin[2]/h));*/
    FLUID::Vec3i bmin_i = floor(bmin / h);
    bmin = h * FLUID::Vec3f(bmin_i);
    bmax = bmin + bulk_size * FLUID::Vec3f(ni, nj, nk);
    fmt::print("2 finish boundary compute\n");
    std::vector<char> buffer;
    std::vector<int> indexBuffer(ni * nj * nk);
#if 0
    std::atomic<int> counter = 0;
    std::vector<std::atomic<char>> abuffer(ni * nj * nk);
    tbb::parallel_for((uint)0, (uint)ni * nj * nk, (uint)1,
                      [&abuffer](uint i) { abuffer[i] = -1; });
    tbb::parallel_for((uint)0, (uint)num_particles, (uint)1, [&](uint i) {
      FLUID::Vec3f pos = particles[i].pos - bmin;
      FLUID::Vec3i bulk_ijk =
          FLUID::Vec3i(floor(pos[0] / bulk_size), floor(pos[1] / bulk_size),
                       floor(pos[2] / bulk_size));
      size_t idx = bulk_ijk[0] + ni * bulk_ijk[1] + ni * nj * bulk_ijk[2];
      char exp = -1;
      if (abuffer[idx].compare_exchange_strong(exp, -2)) {
        auto bulk_id = counter.fetch_add(1);
        indexBuffer[idx] = bulk_id;
      }
    });
    auto cnt = counter.load();
    fluid_bulk.resize(cnt);
    tbb::parallel_for(
        (uint)0, (uint)buffer.size(), (uint)1, [&](uint buffer_id) {
          if (buffer[buffer_id] == 1) {
            int ii = buffer_id % ni, jj = (buffer_id % (ni * nj)) / ni,
                kk = buffer_id / (ni * nj);
            int id = indexBuffer[buffer_id];
            fluid_bulk[id] =
                fluid_Tile<N>(FLUID::Vec3i(N * ii, N * jj, N * kk), id);
          }
        });
#endif
    buffer.resize(ni * nj * nk);
    buffer.assign(ni * nj * nk, 0);
    fmt::print("3 finish hash buffer init\n");
    // abuffer.assign(ni * nj * nk, 0);
    // tbb::parallel_for((uint)0, (uint)ni * nj * nk, (uint)1,
    //                  [&abuffer](uint i) { abuffer[i] = 0; });
      tbb::parallel_for((uint)0, (uint)num_particles, (uint)1, [&](uint i) {
          FLUID::Vec3f pos = particles[i].pos - bmin;
          FLUID::Vec3i bulk_ijk =
                  FLUID::Vec3i(floor(pos[0] / bulk_size), floor(pos[1] / bulk_size),
                               floor(pos[2] / bulk_size));
          for (int kk = bulk_ijk[2] - 0; kk <= bulk_ijk[2] + 0; kk++)
              for (int jj = bulk_ijk[1] - 0; jj <= bulk_ijk[1] + 0; jj++)
                  for (int ii = bulk_ijk[0] - 0; ii <= bulk_ijk[0] + 0; ii++) {
                      if (ii >= 0 && ii < ni && jj >= 0 && jj < nj && kk >= 0 &&
                          kk < nk) {
                          size_t idx = ii + ni * jj + ni * nj * kk;
                          buffer[idx] = 1;
                      }
                  }
      });
      std::vector<char> buffer2(buffer.size());
      buffer2.assign(buffer2.size(),0);
      tbb::parallel_for((uint)0, (uint)ni*nj*nk, (uint)1, [&](uint index) {
          int i=index%ni;
          int j=(index/ni)%nj;
          int k=index/(ni*nj);
          if(buffer[index]==1) {
              for (int kk = k - 2; kk <= k + 2; kk++)
                  for (int jj = j - 2; jj <= j + 2; jj++)
                      for (int ii = i - 2; ii <= i + 2; ii++) {
                          if (ii >= 0 && ii < ni && jj >= 0 && jj < nj && kk >= 0 &&
                              kk < nk) {
                              size_t idx = ii + ni * jj + ni * nj * kk;
                              buffer2[idx] = 1;
                          }
                      }
          }
      });
      buffer = buffer2;
    fmt::print("4 finish hashing\n");

    std::exclusive_scan(std::execution::par, buffer.begin(), buffer.end(),
                        indexBuffer.begin(), 0);
    fmt::print("5 finish scanning\n");

    auto cnt = indexBuffer.back() + buffer.back();
    fluid_bulk.resize(cnt);
    tbb::parallel_for(
        (uint)0, (uint)buffer.size(), (uint)1, [&](uint buffer_id) {
          if (buffer[buffer_id] == 1) {
            int ii = buffer_id % ni, jj = (buffer_id % (ni * nj)) / ni,
                kk = buffer_id / (ni * nj);
            int id = indexBuffer[buffer_id];
            fluid_bulk[id] =
                fluid_Tile<N>(FLUID::Vec3i(N * ii, N * jj, N * kk), id);
          }
        });
    fmt::print("6 finish bulk list build\n");

    for (int buffer_id = 0; buffer_id < fluid_bulk.size(); buffer_id++) {
      FLUID::Vec3i corner = fluid_bulk[buffer_id].tile_corner;
      int ii = corner[0] / N;
      int jj = corner[1] / N;
      int kk = corner[2] / N;
      FLUID::Vec3i bulk_ijk = FLUID::Vec3i(ii, jj, kk);
      index_mapping[bulk_ijk] = buffer_id;
    }
    fmt::print("7 finish bulk mapping build\n");
// int non_boundary_bulks;
#if 0
    for (int i = 0; i < 2; i++) {
      // non_boundary_bulks = fluid_bulk.size();
      extend_fluidbulks();
    }
#endif
    fmt::print("8 extent bulks\n");
    // cout<<non_boundary_bulks<<endl;
    tbb::parallel_for(
        (uint)0, (uint)fluid_bulk.size(), (uint)1, [&](uint buffer_id) {
          FLUID::Vec3i corner = fluid_bulk[buffer_id].tile_corner;
          int ii = corner[0] / N;
          int jj = corner[1] / N;
          int kk = corner[2] / N;
          FLUID::Vec3i bulk_ijk = FLUID::Vec3i(ii, jj, kk);
          if (index_mapping.find(bulk_ijk + FLUID::Vec3i(-1, 0, 0)) ==
                  index_mapping.end() ||
              index_mapping.find(bulk_ijk + FLUID::Vec3i(1, 0, 0)) ==
                  index_mapping.end() ||
              index_mapping.find(bulk_ijk + FLUID::Vec3i(0, 1, 0)) ==
                  index_mapping.end() ||
              index_mapping.find(bulk_ijk + FLUID::Vec3i(0, -1, 0)) ==
                  index_mapping.end() ||
              index_mapping.find(bulk_ijk + FLUID::Vec3i(0, 0, -1)) ==
                  index_mapping.end() ||
              index_mapping.find(bulk_ijk + FLUID::Vec3i(0, 0, 1)) ==
                  index_mapping.end()) {
            fluid_bulk[buffer_id].is_boundary = true;
          }
        });
    fmt::print("9 identify boundary bulks\n");
    n_bulks = fluid_bulk.size();
    cout << "num of bulks:" << fluid_bulk.size() << endl;
    cout << "bmin:" << bmin[0] << " " << bmin[1] << " " << bmin[2] << endl;
    cout << "bmax:" << bmax[0] << " " << bmax[1] << " " << bmax[2] << endl;
    cout << "dimension:" << ni << " " << nj << " " << nk << endl;
  }

  sparse_fluid_3D() {
    n_perbulk = N * N * N;
    fluid_bulk.resize(0);
    tile_n = N;
    chunck3D<char, N> is_there;
    loop_order.resize(0);
    for (int k = 1; k < N - 1; k++)
      for (int j = 1; j < N - 1; j++)
        for (int i = 1; i < N - 1; i++) {
          is_there(i, j, k) = 1;
          loop_order.push_back(FLUID::Vec3i(i, j, k));
        }
    for (int k = 0; k < N; k++)
      for (int j = 0; j < N; j++) {

        if (is_there(0, j, k) == 0) {
          is_there(0, j, k) = 1;
          loop_order.push_back(FLUID::Vec3i(0, j, k));
        }

      } // i==0
    for (int k = 0; k < N; k++)
      for (int j = 0; j < N; j++) {
        if (is_there(N - 1, j, k) == 0) {
          is_there(N - 1, j, k) = 1;
          loop_order.push_back(FLUID::Vec3i(N - 1, j, k));
        }

      } // i==N-1
    for (int k = 0; k < N; k++)
      for (int i = 0; i < N; i++) {
        if (is_there(i, 0, k) == 0) {
          is_there(i, 0, k) = 1;
          loop_order.push_back(FLUID::Vec3i(i, 0, k));
        }

      } // j==0
    for (int k = 0; k < N; k++)
      for (int i = 0; i < N; i++) {
        if (is_there(i, N - 1, k) == 0) {
          is_there(i, N - 1, k) = 1;
          loop_order.push_back(FLUID::Vec3i(i, N - 1, k));
        }
      } // j==N-1
    for (int j = 0; j < N; j++)
      for (int i = 0; i < N; i++) {
        if (is_there(i, j, 0) == 0) {
          is_there(i, j, 0) = 1;
          loop_order.push_back(FLUID::Vec3i(i, j, 0));
        }
      } // k=0
    for (int j = 0; j < N; j++)
      for (int i = 0; i < N; i++) {
        if (is_there(i, j, N - 1) == 0) {
          is_there(i, j, N - 1) = 1;
          loop_order.push_back(FLUID::Vec3i(i, j, N - 1));
        }
      } // k=0

    assert(loop_order.size() == N * N * N);
  }
  ~sparse_fluid_3D() { fluid_bulk.resize(0); }
  void clear() { fluid_bulk.resize(0); }
  int64 find_bulk(int i, int j, int k) {
    int I = i / N, J = j / N, K = k / N;
    int64 bulk_index;
    if (index_mapping.find(FLUID::Vec3i(I, J, K)) == index_mapping.end()) {
      bulk_index = -1;
    } else {
      bulk_index = index_mapping[FLUID::Vec3i(I, J, K)];
    }
    return bulk_index;
  }
  int64 find_bulk(int index, int i, int j, int k) {
    int I = (fluid_bulk[index].tile_corner[0] + i) / N;
    int J = (fluid_bulk[index].tile_corner[1] + j) / N;
    int K = (fluid_bulk[index].tile_corner[2] + k) / N;
    int64 bulk_index;
    if (index_mapping.find(FLUID::Vec3i(I, J, K)) == index_mapping.end()) {
      bulk_index = -1;
    } else {
      bulk_index = index_mapping[FLUID::Vec3i(I, J, K)];
    }
    return bulk_index;
  }
  float &omega_x(int64 bulk_index, int i, int j, int k) {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).omega_x(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].omega_x((i + N) % N, (j + N) % N,
                                               (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].omega_x(min(max(0, i), N - 1),
                                              min(max(0, j), N - 1),
                                              min(max(0, k), N - 1));
      }
    }
  }
  float const &omega_x(int64 bulk_index, int i, int j, int k) const {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).omega_x(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].omega_x((i + N) % N, (j + N) % N,
                                               (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].omega_x(min(max(0, i), N - 1),
                                              min(max(0, j), N - 1),
                                              min(max(0, k), N - 1));
      }
    }
  }

  float &omega_y(int64 bulk_index, int i, int j, int k) {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).omega_y(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].omega_y((i + N) % N, (j + N) % N,
                                               (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].omega_y(min(max(0, i), N - 1),
                                              min(max(0, j), N - 1),
                                              min(max(0, k), N - 1));
      }
    }
  }
  float const &omega_y(int64 bulk_index, int i, int j, int k) const {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).omega_y(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].omega_y((i + N) % N, (j + N) % N,
                                               (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].omega_y(min(max(0, i), N - 1),
                                              min(max(0, j), N - 1),
                                              min(max(0, k), N - 1));
      }
    }
  }

  float &omega_z(int64 bulk_index, int i, int j, int k) {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).omega_z(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].omega_z((i + N) % N, (j + N) % N,
                                               (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].omega_z(min(max(0, i), N - 1),
                                              min(max(0, j), N - 1),
                                              min(max(0, k), N - 1));
      }
    }
  }
  float const &omega_z(int64 bulk_index, int i, int j, int k) const {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).omega_z(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].omega_z((i + N) % N, (j + N) % N,
                                               (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].omega_z(min(max(0, i), N - 1),
                                              min(max(0, j), N - 1),
                                              min(max(0, k), N - 1));
      }
    }
  }

  float &omega_x_save(int64 bulk_index, int i, int j, int k) {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).omega_x_save(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].omega_x_save((i + N) % N, (j + N) % N,
                                                    (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].omega_x_save(min(max(0, i), N - 1),
                                                   min(max(0, j), N - 1),
                                                   min(max(0, k), N - 1));
      }
    }
  }
  float const &omega_x_save(int64 bulk_index, int i, int j, int k) const {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).omega_x_save(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].omega_x_save((i + N) % N, (j + N) % N,
                                                    (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].omega_x_save(min(max(0, i), N - 1),
                                                   min(max(0, j), N - 1),
                                                   min(max(0, k), N - 1));
      }
    }
  }

  float &omega_y_save(int64 bulk_index, int i, int j, int k) {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).omega_y_save(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].omega_y_save((i + N) % N, (j + N) % N,
                                                    (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].omega_y_save(min(max(0, i), N - 1),
                                                   min(max(0, j), N - 1),
                                                   min(max(0, k), N - 1));
      }
    }
  }
  float const &omega_y_save(int64 bulk_index, int i, int j, int k) const {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).omega_y_save(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].omega_y_save((i + N) % N, (j + N) % N,
                                                    (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].omega_y_save(min(max(0, i), N - 1),
                                                   min(max(0, j), N - 1),
                                                   min(max(0, k), N - 1));
      }
    }
  }

  float &omega_z_save(int64 bulk_index, int i, int j, int k) {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).omega_z_save(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].omega_z_save((i + N) % N, (j + N) % N,
                                                    (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].omega_z_save(min(max(0, i), N - 1),
                                                   min(max(0, j), N - 1),
                                                   min(max(0, k), N - 1));
      }
    }
  }
  float const &omega_z_save(int64 bulk_index, int i, int j, int k) const {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).omega_z_save(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].omega_z_save((i + N) % N, (j + N) % N,
                                                    (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].omega_z_save(min(max(0, i), N - 1),
                                                   min(max(0, j), N - 1),
                                                   min(max(0, k), N - 1));
      }
    }
  }

  float &omega_x_delta(int64 bulk_index, int i, int j, int k) {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).omega_x_delta(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].omega_x_delta((i + N) % N, (j + N) % N,
                                                     (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].omega_x_delta(min(max(0, i), N - 1),
                                                    min(max(0, j), N - 1),
                                                    min(max(0, k), N - 1));
      }
    }
  }
  float const &omega_x_delta(int64 bulk_index, int i, int j, int k) const {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).omega_x_delta(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].omega_x_delta((i + N) % N, (j + N) % N,
                                                     (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].omega_x_delta(min(max(0, i), N - 1),
                                                    min(max(0, j), N - 1),
                                                    min(max(0, k), N - 1));
      }
    }
  }

  float &omega_y_delta(int64 bulk_index, int i, int j, int k) {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).omega_y_delta(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].omega_y_delta((i + N) % N, (j + N) % N,
                                                     (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].omega_y_delta(min(max(0, i), N - 1),
                                                    min(max(0, j), N - 1),
                                                    min(max(0, k), N - 1));
      }
    }
  }
  float const &omega_y_delta(int64 bulk_index, int i, int j, int k) const {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).omega_y_delta(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].omega_y_delta((i + N) % N, (j + N) % N,
                                                     (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].omega_y_delta(min(max(0, i), N - 1),
                                                    min(max(0, j), N - 1),
                                                    min(max(0, k), N - 1));
      }
    }
  }

  float &omega_z_delta(int64 bulk_index, int i, int j, int k) {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).omega_z_delta(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].omega_z_delta((i + N) % N, (j + N) % N,
                                                     (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].omega_z_delta(min(max(0, i), N - 1),
                                                    min(max(0, j), N - 1),
                                                    min(max(0, k), N - 1));
      }
    }
  }
  float const &omega_z_delta(int64 bulk_index, int i, int j, int k) const {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).omega_z_delta(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].omega_z_delta((i + N) % N, (j + N) % N,
                                                     (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].omega_z_delta(min(max(0, i), N - 1),
                                                    min(max(0, j), N - 1),
                                                    min(max(0, k), N - 1));
      }
    }
  }

  float &psi_x(int64 bulk_index, int i, int j, int k) {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).psi_x(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].psi_x((i + N) % N, (j + N) % N,
                                             (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].psi_x(min(max(0, i), N - 1),
                                            min(max(0, j), N - 1),
                                            min(max(0, k), N - 1));
      }
    }
  }
  float const &psi_x(int64 bulk_index, int i, int j, int k) const {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).psi_x(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].psi_x((i + N) % N, (j + N) % N,
                                             (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].psi_x(min(max(0, i), N - 1),
                                            min(max(0, j), N - 1),
                                            min(max(0, k), N - 1));
      }
    }
  }

  float &psi_y(int64 bulk_index, int i, int j, int k) {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).psi_y(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].psi_y((i + N) % N, (j + N) % N,
                                             (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].psi_y(min(max(0, i), N - 1),
                                            min(max(0, j), N - 1),
                                            min(max(0, k), N - 1));
      }
    }
  }
  float const &psi_y(int64 bulk_index, int i, int j, int k) const {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).psi_y(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].psi_y((i + N) % N, (j + N) % N,
                                             (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].psi_y(min(max(0, i), N - 1),
                                            min(max(0, j), N - 1),
                                            min(max(0, k), N - 1));
      }
    }
  }

  float &psi_z(int64 bulk_index, int i, int j, int k) {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).psi_z(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].psi_z((i + N) % N, (j + N) % N,
                                             (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].psi_z(min(max(0, i), N - 1),
                                            min(max(0, j), N - 1),
                                            min(max(0, k), N - 1));
      }
    }
  }
  float const &psi_z(int64 bulk_index, int i, int j, int k) const {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).psi_z(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].psi_z((i + N) % N, (j + N) % N,
                                             (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].psi_z(min(max(0, i), N - 1),
                                            min(max(0, j), N - 1),
                                            min(max(0, k), N - 1));
      }
    }
  }

  float &u(int64 bulk_index, int i, int j, int k) {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).u(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].u((i + N) % N, (j + N) % N, (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].u(min(max(0, i), N - 1),
                                        min(max(0, j), N - 1),
                                        min(max(0, k), N - 1));
      }
    }
  }
  float const &u(int64 bulk_index, int i, int j, int k) const {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).u(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].u((i + N) % N, (j + N) % N, (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].u(min(max(0, i), N - 1),
                                        min(max(0, j), N - 1),
                                        min(max(0, k), N - 1));
      }
    }
  }

  float &u_save(int64 bulk_index, int i, int j, int k) {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).u_save(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].u_save((i + N) % N, (j + N) % N,
                                              (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].u_save(min(max(0, i), N - 1),
                                             min(max(0, j), N - 1),
                                             min(max(0, k), N - 1));
      }
    }
  }
  float const &u_save(int64 bulk_index, int i, int j, int k) const {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).u_save(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].u_save((i + N) % N, (j + N) % N,
                                              (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].u_save(min(max(0, i), N - 1),
                                             min(max(0, j), N - 1),
                                             min(max(0, k), N - 1));
      }
    }
  }

  float &u_delta(int64 bulk_index, int i, int j, int k) {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).u_delta(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].u_delta((i + N) % N, (j + N) % N,
                                               (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].u_delta(min(max(0, i), N - 1),
                                              min(max(0, j), N - 1),
                                              min(max(0, k), N - 1));
      }
    }
  }
  float const &u_delta(int64 bulk_index, int i, int j, int k) const {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).u_delta(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].u_delta((i + N) % N, (j + N) % N,
                                               (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].u_delta(min(max(0, i), N - 1),
                                              min(max(0, j), N - 1),
                                              min(max(0, k), N - 1));
      }
    }
  }

  float &u_coef(int64 bulk_index, int i, int j, int k) {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).u_coef(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].u_coef((i + N) % N, (j + N) % N,
                                              (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].u_coef(min(max(0, i), N - 1),
                                             min(max(0, j), N - 1),
                                             min(max(0, k), N - 1));
      }
    }
  }
  float const &u_coef(int64 bulk_index, int i, int j, int k) const {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).u_coef(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].u_coef((i + N) % N, (j + N) % N,
                                              (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].u_coef(min(max(0, i), N - 1),
                                             min(max(0, j), N - 1),
                                             min(max(0, k), N - 1));
      }
    }
  }

  float &u_extrapolate(int64 bulk_index, int i, int j, int k) {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).u_extrapolate(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].u_extrapolate((i + N) % N, (j + N) % N,
                                                     (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].u_extrapolate(min(max(0, i), N - 1),
                                                    min(max(0, j), N - 1),
                                                    min(max(0, k), N - 1));
      }
    }
  }
  float const &u_extrapolate(int64 bulk_index, int i, int j, int k) const {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).u_extrapolate(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].u_extrapolate((i + N) % N, (j + N) % N,
                                                     (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].u_extrapolate(min(max(0, i), N - 1),
                                                    min(max(0, j), N - 1),
                                                    min(max(0, k), N - 1));
      }
    }
  }

  float &v(int64 bulk_index, int i, int j, int k) {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).v(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].v((i + N) % N, (j + N) % N, (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].v(min(max(0, i), N - 1),
                                        min(max(0, j), N - 1),
                                        min(max(0, k), N - 1));
      }
    }
  }
  float const &v(int64 bulk_index, int i, int j, int k) const {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).v(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].v((i + N) % N, (j + N) % N, (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].v(min(max(0, i), N - 1),
                                        min(max(0, j), N - 1),
                                        min(max(0, k), N - 1));
      }
    }
  }

  float &v_save(int64 bulk_index, int i, int j, int k) {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).v_save(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].v_save((i + N) % N, (j + N) % N,
                                              (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].v_save(min(max(0, i), N - 1),
                                             min(max(0, j), N - 1),
                                             min(max(0, k), N - 1));
      }
    }
  }
  float const &v_save(int64 bulk_index, int i, int j, int k) const {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).v_save(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].v_save((i + N) % N, (j + N) % N,
                                              (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].v_save(min(max(0, i), N - 1),
                                             min(max(0, j), N - 1),
                                             min(max(0, k), N - 1));
      }
    }
  }

  float &v_delta(int64 bulk_index, int i, int j, int k) {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).v_delta(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].v_delta((i + N) % N, (j + N) % N,
                                               (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].v_delta(min(max(0, i), N - 1),
                                              min(max(0, j), N - 1),
                                              min(max(0, k), N - 1));
      }
    }
  }
  float const &v_delta(int64 bulk_index, int i, int j, int k) const {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).v_delta(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].v_delta((i + N) % N, (j + N) % N,
                                               (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].v_delta(min(max(0, i), N - 1),
                                              min(max(0, j), N - 1),
                                              min(max(0, k), N - 1));
      }
    }
  }

  float &v_coef(int64 bulk_index, int i, int j, int k) {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).v_coef(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].v_coef((i + N) % N, (j + N) % N,
                                              (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].v_coef(min(max(0, i), N - 1),
                                             min(max(0, j), N - 1),
                                             min(max(0, k), N - 1));
      }
    }
  }
  float const &v_coef(int64 bulk_index, int i, int j, int k) const {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).v_coef(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].v_coef((i + N) % N, (j + N) % N,
                                              (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].v_coef(min(max(0, i), N - 1),
                                             min(max(0, j), N - 1),
                                             min(max(0, k), N - 1));
      }
    }
  }

  float &v_extrapolate(int64 bulk_index, int i, int j, int k) {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).v_extrapolate(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].v_extrapolate((i + N) % N, (j + N) % N,
                                                     (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].v_extrapolate(min(max(0, i), N - 1),
                                                    min(max(0, j), N - 1),
                                                    min(max(0, k), N - 1));
      }
    }
  }
  float const &v_extrapolate(int64 bulk_index, int i, int j, int k) const {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).v_extrapolate(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].v_extrapolate((i + N) % N, (j + N) % N,
                                                     (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].v_extrapolate(min(max(0, i), N - 1),
                                                    min(max(0, j), N - 1),
                                                    min(max(0, k), N - 1));
      }
    }
  }

  float &w(int64 bulk_index, int i, int j, int k) {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).w(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].w((i + N) % N, (j + N) % N, (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].w(min(max(0, i), N - 1),
                                        min(max(0, j), N - 1),
                                        min(max(0, k), N - 1));
      }
    }
  }
  float const &w(int64 bulk_index, int i, int j, int k) const {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).w(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].w((i + N) % N, (j + N) % N, (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].w(min(max(0, i), N - 1),
                                        min(max(0, j), N - 1),
                                        min(max(0, k), N - 1));
      }
    }
  }

  float &w_save(int64 bulk_index, int i, int j, int k) {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).w_save(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].w_save((i + N) % N, (j + N) % N,
                                              (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].w_save(min(max(0, i), N - 1),
                                             min(max(0, j), N - 1),
                                             min(max(0, k), N - 1));
      }
    }
  }
  float const &w_save(int64 bulk_index, int i, int j, int k) const {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).w_save(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].w_save((i + N) % N, (j + N) % N,
                                              (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].w_save(min(max(0, i), N - 1),
                                             min(max(0, j), N - 1),
                                             min(max(0, k), N - 1));
      }
    }
  }

  float &w_delta(int64 bulk_index, int i, int j, int k) {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).w_delta(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].w_delta((i + N) % N, (j + N) % N,
                                               (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].w_delta(min(max(0, i), N - 1),
                                              min(max(0, j), N - 1),
                                              min(max(0, k), N - 1));
      }
    }
  }
  float const &w_delta(int64 bulk_index, int i, int j, int k) const {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).w_delta(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].w_delta((i + N) % N, (j + N) % N,
                                               (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].w_delta(min(max(0, i), N - 1),
                                              min(max(0, j), N - 1),
                                              min(max(0, k), N - 1));
      }
    }
  }

  float &w_coef(int64 bulk_index, int i, int j, int k) {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).w_coef(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].w_coef((i + N) % N, (j + N) % N,
                                              (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].w_coef(min(max(0, i), N - 1),
                                             min(max(0, j), N - 1),
                                             min(max(0, k), N - 1));
      }
    }
  }
  float const &w_coef(int64 bulk_index, int i, int j, int k) const {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).w_coef(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].w_coef((i + N) % N, (j + N) % N,
                                              (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].w_coef(min(max(0, i), N - 1),
                                             min(max(0, j), N - 1),
                                             min(max(0, k), N - 1));
      }
    }
  }

  float &w_extrapolate(int64 bulk_index, int i, int j, int k) {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).w_extrapolate(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].w_extrapolate((i + N) % N, (j + N) % N,
                                                     (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].u_extrapolate(min(max(0, i), N - 1),
                                                    min(max(0, j), N - 1),
                                                    min(max(0, k), N - 1));
      }
    }
  }
  float const &w_extrapolate(int64 bulk_index, int i, int j, int k) const {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).w_extrapolate(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].w_extrapolate((i + N) % N, (j + N) % N,
                                                     (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].w_extrapolate(min(max(0, i), N - 1),
                                                    min(max(0, j), N - 1),
                                                    min(max(0, k), N - 1));
      }
    }
  }

  float &liquid_phi(int64 bulk_index, int i, int j, int k) {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).liquid_phi(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].liquid_phi((i + N) % N, (j + N) % N,
                                                  (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].liquid_phi(min(max(0, i), N - 1),
                                                 min(max(0, j), N - 1),
                                                 min(max(0, k), N - 1));
      }
    }
  }
  float const &liquid_phi(int64 bulk_index, int i, int j, int k) const {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).liquid_phi(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].liquid_phi((i + N) % N, (j + N) % N,
                                                  (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].liquid_phi(min(max(0, i), N - 1),
                                                 min(max(0, j), N - 1),
                                                 min(max(0, k), N - 1));
      }
    }
  }

  float &solid_phi(int64 bulk_index, int i, int j, int k) {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).solid_phi(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].solid_phi((i + N) % N, (j + N) % N,
                                                 (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].solid_phi(min(max(0, i), N - 1),
                                                min(max(0, j), N - 1),
                                                min(max(0, k), N - 1));
      }
    }
  }
  float const &solid_phi(int64 bulk_index, int i, int j, int k) const {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).solid_phi(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].solid_phi((i + N) % N, (j + N) % N,
                                                 (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].solid_phi(min(max(0, i), N - 1),
                                                min(max(0, j), N - 1),
                                                min(max(0, k), N - 1));
      }
    }
  }

  float &u_weight(int64 bulk_index, int i, int j, int k) {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).u_weight(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].u_weight((i + N) % N, (j + N) % N,
                                                (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].u_weight(min(max(0, i), N - 1),
                                               min(max(0, j), N - 1),
                                               min(max(0, k), N - 1));
      }
    }
  }
  float const &u_weight(int64 bulk_index, int i, int j, int k) const {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).u_weight(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].u_weight((i + N) % N, (j + N) % N,
                                                (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].u_weight(min(max(0, i), N - 1),
                                               min(max(0, j), N - 1),
                                               min(max(0, k), N - 1));
      }
    }
  }
  float &v_weight(int64 bulk_index, int i, int j, int k) {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).v_weight(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].v_weight((i + N) % N, (j + N) % N,
                                                (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].v_weight(min(max(0, i), N - 1),
                                               min(max(0, j), N - 1),
                                               min(max(0, k), N - 1));
      }
    }
  }
  float const &v_weight(int64 bulk_index, int i, int j, int k) const {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).v_weight(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].v_weight((i + N) % N, (j + N) % N,
                                                (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].v_weight(min(max(0, i), N - 1),
                                               min(max(0, j), N - 1),
                                               min(max(0, k), N - 1));
      }
    }
  }
  float &w_weight(int64 bulk_index, int i, int j, int k) {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).w_weight(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].w_weight((i + N) % N, (j + N) % N,
                                                (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].w_weight(min(max(0, i), N - 1),
                                               min(max(0, j), N - 1),
                                               min(max(0, k), N - 1));
      }
    }
  }
  float const &w_weight(int64 bulk_index, int i, int j, int k) const {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).w_weight(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].w_weight((i + N) % N, (j + N) % N,
                                                (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].w_weight(min(max(0, i), N - 1),
                                               min(max(0, j), N - 1),
                                               min(max(0, k), N - 1));
      }
    }
  }

  float &u_solid(int64 bulk_index, int i, int j, int k) {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).u_solid(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].u_solid((i + N) % N, (j + N) % N,
                                               (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].u_solid(min(max(0, i), N - 1),
                                              min(max(0, j), N - 1),
                                              min(max(0, k), N - 1));
      }
    }
  }
  float const &u_solid(int64 bulk_index, int i, int j, int k) const {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).u_solid(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].u_solid((i + N) % N, (j + N) % N,
                                               (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].u_solid(min(max(0, i), N - 1),
                                              min(max(0, j), N - 1),
                                              min(max(0, k), N - 1));
      }
    }
  }
  float &v_solid(int64 bulk_index, int i, int j, int k) {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).v_solid(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].v_solid((i + N) % N, (j + N) % N,
                                               (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].v_solid(min(max(0, i), N - 1),
                                              min(max(0, j), N - 1),
                                              min(max(0, k), N - 1));
      }
    }
  }
  float const &v_solid(int64 bulk_index, int i, int j, int k) const {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).v_solid(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].v_solid((i + N) % N, (j + N) % N,
                                               (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].v_solid(min(max(0, i), N - 1),
                                              min(max(0, j), N - 1),
                                              min(max(0, k), N - 1));
      }
    }
  }
  float &w_solid(int64 bulk_index, int i, int j, int k) {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).w_solid(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].w_solid((i + N) % N, (j + N) % N,
                                               (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].w_solid(min(max(0, i), N - 1),
                                              min(max(0, j), N - 1),
                                              min(max(0, k), N - 1));
      }
    }
  }
  float const &w_solid(int64 bulk_index, int i, int j, int k) const {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).w_solid(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].w_solid((i + N) % N, (j + N) % N,
                                               (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].w_solid(min(max(0, i), N - 1),
                                              min(max(0, j), N - 1),
                                              min(max(0, k), N - 1));
      }
    }
  }

  char &u_valid(int64 bulk_index, int i, int j, int k) {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).u_valid(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].u_valid((i + N) % N, (j + N) % N,
                                               (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].u_valid(min(max(0, i), N - 1),
                                              min(max(0, j), N - 1),
                                              min(max(0, k), N - 1));
      }
    }
  }
  char const &u_valid(int64 bulk_index, int i, int j, int k) const {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).u_valid(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].u_valid((i + N) % N, (j + N) % N,
                                               (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].u_valid(min(max(0, i), N - 1),
                                              min(max(0, j), N - 1),
                                              min(max(0, k), N - 1));
      }
    }
  }
  char &v_valid(int64 bulk_index, int i, int j, int k) {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).v_valid(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].v_valid((i + N) % N, (j + N) % N,
                                               (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].v_valid(min(max(0, i), N - 1),
                                              min(max(0, j), N - 1),
                                              min(max(0, k), N - 1));
      }
    }
  }
  char const &v_valid(int64 bulk_index, int i, int j, int k) const {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).v_valid(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].v_valid((i + N) % N, (j + N) % N,
                                               (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].v_valid(min(max(0, i), N - 1),
                                              min(max(0, j), N - 1),
                                              min(max(0, k), N - 1));
      }
    }
  }
  char &w_valid(int64 bulk_index, int i, int j, int k) {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).w_valid(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].w_valid((i + N) % N, (j + N) % N,
                                               (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].w_valid(min(max(0, i), N - 1),
                                              min(max(0, j), N - 1),
                                              min(max(0, k), N - 1));
      }
    }
  }
  char const &w_valid(int64 bulk_index, int i, int j, int k) const {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).w_valid(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].w_valid((i + N) % N, (j + N) % N,
                                               (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].w_valid(min(max(0, i), N - 1),
                                              min(max(0, j), N - 1),
                                              min(max(0, k), N - 1));
      }
    }
  }

  char &old_valid(int64 bulk_index, int i, int j, int k) {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).old_valid(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].old_valid((i + N) % N, (j + N) % N,
                                                 (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].old_valid(min(max(0, i), N - 1),
                                                min(max(0, j), N - 1),
                                                min(max(0, k), N - 1));
      }
    }
  }
  char const &old_valid(int64 bulk_index, int i, int j, int k) const {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).old_valid(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].old_valid((i + N) % N, (j + N) % N,
                                                 (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].old_valid(min(max(0, i), N - 1),
                                                min(max(0, j), N - 1),
                                                min(max(0, k), N - 1));
      }
    }
  }

  double &pressure(int64 bulk_index, int i, int j, int k) {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).pressure(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].pressure((i + N) % N, (j + N) % N,
                                                (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].pressure(min(max(0, i), N - 1),
                                               min(max(0, j), N - 1),
                                               min(max(0, k), N - 1));
      }
    }
  }
  double const &pressure(int64 bulk_index, int i, int j, int k) const {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).pressure(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].pressure((i + N) % N, (j + N) % N,
                                                (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].pressure(min(max(0, i), N - 1),
                                               min(max(0, j), N - 1),
                                               min(max(0, k), N - 1));
      }
    }
  }

  uint &global_index(int64 bulk_index, int i, int j, int k) {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).global_index(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].global_index((i + N) % N, (j + N) % N,
                                                    (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].global_index(min(max(0, i), N - 1),
                                                   min(max(0, j), N - 1),
                                                   min(max(0, k), N - 1));
      }
    }
  }
  uint const &global_index(int64 bulk_index, int i, int j, int k) const {
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
      return (fluid_bulk[bulk_index]).global_index(i, j, k);
    } else {
      int64 bulk_index2 = find_bulk(fluid_bulk[bulk_index].tile_corner[0] + i,
                                    fluid_bulk[bulk_index].tile_corner[1] + j,
                                    fluid_bulk[bulk_index].tile_corner[2] + k);
      if (bulk_index2 != -1) {
        return fluid_bulk[bulk_index2].global_index((i + N) % N, (j + N) % N,
                                                    (k + N) % N);
      } else {
        return fluid_bulk[bulk_index].global_index(min(max(0, i), N - 1),
                                                   min(max(0, j), N - 1),
                                                   min(max(0, k), N - 1));
      }
    }
  }
  void linear_coef(FLUID::Vec3f &pos, int &i, int &j, int &k, float &fx,
                   float &fy, float &fz) {
    i = floor(pos[0] / h);
    j = floor(pos[1] / h);
    k = floor(pos[2] / h);
    fx = pos[0] / h - (float)i;
    fy = pos[1] / h - (float)j;
    fz = pos[2] / h - (float)k;
  }

  inline bool isValid_u(FLUID::Vec3f &pos) {
    FLUID::Vec3f local_pos = pos - FLUID::Vec3f(0, 0.5 * h, 0.5 * h) - bmin;
    int i, j, k;
    float fx, fy, fz;
    linear_coef(local_pos, i, j, k, fx, fy, fz);
    int64 bulk_idx = find_bulk(i, j, k);
    if (bulk_idx != -1) {
      int local_i = i % N;
      int local_j = j % N;
      int local_k = k % N;
      char v000 = u_valid(bulk_idx, local_i, local_j, local_k);
      char v001 = u_valid(bulk_idx, local_i + 1, local_j, local_k);
      char v010 = u_valid(bulk_idx, local_i, local_j + 1, local_k);
      char v011 = u_valid(bulk_idx, local_i + 1, local_j + 1, local_k);
      char v100 = u_valid(bulk_idx, local_i, local_j, local_k + 1);
      char v101 = u_valid(bulk_idx, local_i + 1, local_j, local_k + 1);
      char v110 = u_valid(bulk_idx, local_i, local_j + 1, local_k + 1);
      char v111 = u_valid(bulk_idx, local_i + 1, local_j + 1, local_k + 1);
      return v000 && v001 && v010 && v011 && v100 && v101 && v110 && v111;
    } else {
      return false;
    }
  }
  inline bool isValid_v(FLUID::Vec3f &pos) {
    FLUID::Vec3f local_pos = pos - FLUID::Vec3f(0.5 * h, 0.0, 0.5 * h) - bmin;
    int i, j, k;
    float fx, fy, fz;
    linear_coef(local_pos, i, j, k, fx, fy, fz);
    int64 bulk_idx = find_bulk(i, j, k);
    if (bulk_idx != -1) {
      int local_i = i % N;
      int local_j = j % N;
      int local_k = k % N;
      char v000 = v_valid(bulk_idx, local_i, local_j, local_k);
      char v001 = v_valid(bulk_idx, local_i + 1, local_j, local_k);
      char v010 = v_valid(bulk_idx, local_i, local_j + 1, local_k);
      char v011 = v_valid(bulk_idx, local_i + 1, local_j + 1, local_k);
      char v100 = v_valid(bulk_idx, local_i, local_j, local_k + 1);
      char v101 = v_valid(bulk_idx, local_i + 1, local_j, local_k + 1);
      char v110 = v_valid(bulk_idx, local_i, local_j + 1, local_k + 1);
      char v111 = v_valid(bulk_idx, local_i + 1, local_j + 1, local_k + 1);
      return v000 && v001 && v010 && v011 && v100 && v101 && v110 && v111;
    } else {
      return false;
    }
  }
  inline bool isValid_w(FLUID::Vec3f &pos) {
    FLUID::Vec3f local_pos = pos - FLUID::Vec3f(0.5 * h, 0.5 * h, 0.0) - bmin;
    int i, j, k;
    float fx, fy, fz;
    linear_coef(local_pos, i, j, k, fx, fy, fz);
    int64 bulk_idx = find_bulk(i, j, k);
    if (bulk_idx != -1) {
      int local_i = i % N;
      int local_j = j % N;
      int local_k = k % N;
      char v000 = w_valid(bulk_idx, local_i, local_j, local_k);
      char v001 = w_valid(bulk_idx, local_i + 1, local_j, local_k);
      char v010 = w_valid(bulk_idx, local_i, local_j + 1, local_k);
      char v011 = w_valid(bulk_idx, local_i + 1, local_j + 1, local_k);
      char v100 = w_valid(bulk_idx, local_i, local_j, local_k + 1);
      char v101 = w_valid(bulk_idx, local_i + 1, local_j, local_k + 1);
      char v110 = w_valid(bulk_idx, local_i, local_j + 1, local_k + 1);
      char v111 = w_valid(bulk_idx, local_i + 1, local_j + 1, local_k + 1);
      return v000 && v001 && v010 && v011 && v100 && v101 && v110 && v111;
    } else {
      return false;
    }
  }
  bool isValidVel(FLUID::Vec3f &pos) {
    return isValid_u(pos) && isValid_v(pos) && isValid_w(pos);
  }
  bool isIsolated(FLUID::Vec3f &pos) {
    FLUID::Vec3f local_pos =
        pos - FLUID::Vec3f(0.5 * h, 0.5 * h, 0.5 * h) - bmin;
    int i, j, k;
    float fx, fy, fz;
    linear_coef(local_pos, i, j, k, fx, fy, fz);
    int64 bulk_idx = find_bulk(i, j, k);
    if (bulk_idx != -1) {
      int local_i = i % N;
      int local_j = j % N;
      int local_k = k % N;
      for (int kk = local_k - 1; kk <= local_k + 1; kk++)
        for (int jj = local_j - 1; jj <= local_j + 1; jj++)
          for (int ii = local_i - 1; ii <= local_i + 1; ii++) {
            if (liquid_phi(bulk_idx, ii, jj, kk) < 0)
              return false;
          }
      return true;
    } else {
      return true;
    }
  }
  float get_u(FLUID::Vec3f &pos) {
    FLUID::Vec3f local_pos = pos - FLUID::Vec3f(0, 0.5 * h, 0.5 * h) - bmin;
    int i, j, k;
    float fx, fy, fz;
    linear_coef(local_pos, i, j, k, fx, fy, fz);
    int64 bulk_idx = find_bulk(i, j, k);
    if (bulk_idx != -1) {
      int local_i = i % N;
      int local_j = j % N;
      int local_k = k % N;
      float v000 = u(bulk_idx, local_i, local_j, local_k);
      float v001 = u(bulk_idx, local_i + 1, local_j, local_k);
      float v010 = u(bulk_idx, local_i, local_j + 1, local_k);
      float v011 = u(bulk_idx, local_i + 1, local_j + 1, local_k);
      float v100 = u(bulk_idx, local_i, local_j, local_k + 1);
      float v101 = u(bulk_idx, local_i + 1, local_j, local_k + 1);
      float v110 = u(bulk_idx, local_i, local_j + 1, local_k + 1);
      float v111 = u(bulk_idx, local_i + 1, local_j + 1, local_k + 1);
      return trilerp(v000, v001, v010, v011, v100, v101, v110, v111, fx, fy,
                     fz);
    } else {
      return 0;
    }
  }
  float get_v(FLUID::Vec3f &pos) {
    FLUID::Vec3f local_pos = pos - FLUID::Vec3f(0.5 * h, 0, 0.5 * h) - bmin;
    int i, j, k;
    float fx, fy, fz;
    linear_coef(local_pos, i, j, k, fx, fy, fz);
    int64 bulk_idx = find_bulk(i, j, k);
    if (bulk_idx != -1) {
      int local_i = i % N;
      int local_j = j % N;
      int local_k = k % N;
      float v000 = v(bulk_idx, local_i, local_j, local_k);
      float v001 = v(bulk_idx, local_i + 1, local_j, local_k);
      float v010 = v(bulk_idx, local_i, local_j + 1, local_k);
      float v011 = v(bulk_idx, local_i + 1, local_j + 1, local_k);
      float v100 = v(bulk_idx, local_i, local_j, local_k + 1);
      float v101 = v(bulk_idx, local_i + 1, local_j, local_k + 1);
      float v110 = v(bulk_idx, local_i, local_j + 1, local_k + 1);
      float v111 = v(bulk_idx, local_i + 1, local_j + 1, local_k + 1);
      return trilerp(v000, v001, v010, v011, v100, v101, v110, v111, fx, fy,
                     fz);
    } else {
      return 0;
    }
  }
  float get_w(FLUID::Vec3f &pos) {
    FLUID::Vec3f local_pos = pos - FLUID::Vec3f(0.5 * h, 0.5 * h, 0) - bmin;
    int i, j, k;
    float fx, fy, fz;
    linear_coef(local_pos, i, j, k, fx, fy, fz);
    int64 bulk_idx = find_bulk(i, j, k);
    if (bulk_idx != -1) {
      int local_i = i % N;
      int local_j = j % N;
      int local_k = k % N;
      float v000 = w(bulk_idx, local_i, local_j, local_k);
      float v001 = w(bulk_idx, local_i + 1, local_j, local_k);
      float v010 = w(bulk_idx, local_i, local_j + 1, local_k);
      float v011 = w(bulk_idx, local_i + 1, local_j + 1, local_k);
      float v100 = w(bulk_idx, local_i, local_j, local_k + 1);
      float v101 = w(bulk_idx, local_i + 1, local_j, local_k + 1);
      float v110 = w(bulk_idx, local_i, local_j + 1, local_k + 1);
      float v111 = w(bulk_idx, local_i + 1, local_j + 1, local_k + 1);
      return trilerp(v000, v001, v010, v011, v100, v101, v110, v111, fx, fy,
                     fz);
    } else {
      return 0;
    }
  }
  float get_du(FLUID::Vec3f &pos) {
    FLUID::Vec3f local_pos = pos - FLUID::Vec3f(0, 0.5 * h, 0.5 * h) - bmin;
    int i, j, k;
    float fx, fy, fz;
    linear_coef(local_pos, i, j, k, fx, fy, fz);
    int64 bulk_idx = find_bulk(i, j, k);
    if (bulk_idx != -1) {
      int local_i = i % N;
      int local_j = j % N;
      int local_k = k % N;
      float v000 = u_delta(bulk_idx, local_i, local_j, local_k);
      float v001 = u_delta(bulk_idx, local_i + 1, local_j, local_k);
      float v010 = u_delta(bulk_idx, local_i, local_j + 1, local_k);
      float v011 = u_delta(bulk_idx, local_i + 1, local_j + 1, local_k);
      float v100 = u_delta(bulk_idx, local_i, local_j, local_k + 1);
      float v101 = u_delta(bulk_idx, local_i + 1, local_j, local_k + 1);
      float v110 = u_delta(bulk_idx, local_i, local_j + 1, local_k + 1);
      float v111 = u_delta(bulk_idx, local_i + 1, local_j + 1, local_k + 1);
      return trilerp(v000, v001, v010, v011, v100, v101, v110, v111, fx, fy,
                     fz);
    } else {
      return 0;
    }
  }
  float get_dv(FLUID::Vec3f &pos) {
    FLUID::Vec3f local_pos = pos - FLUID::Vec3f(0.5 * h, 0, 0.5 * h) - bmin;
    int i, j, k;
    float fx, fy, fz;
    linear_coef(local_pos, i, j, k, fx, fy, fz);
    int64 bulk_idx = find_bulk(i, j, k);
    if (bulk_idx != -1) {
      int local_i = i % N;
      int local_j = j % N;
      int local_k = k % N;
      float v000 = v_delta(bulk_idx, local_i, local_j, local_k);
      float v001 = v_delta(bulk_idx, local_i + 1, local_j, local_k);
      float v010 = v_delta(bulk_idx, local_i, local_j + 1, local_k);
      float v011 = v_delta(bulk_idx, local_i + 1, local_j + 1, local_k);
      float v100 = v_delta(bulk_idx, local_i, local_j, local_k + 1);
      float v101 = v_delta(bulk_idx, local_i + 1, local_j, local_k + 1);
      float v110 = v_delta(bulk_idx, local_i, local_j + 1, local_k + 1);
      float v111 = v_delta(bulk_idx, local_i + 1, local_j + 1, local_k + 1);
      return trilerp(v000, v001, v010, v011, v100, v101, v110, v111, fx, fy,
                     fz);
    } else {
      return 0;
    }
  }
  float get_dw(FLUID::Vec3f &pos) {
    FLUID::Vec3f local_pos = pos - FLUID::Vec3f(0.5 * h, 0.5 * h, 0) - bmin;
    int i, j, k;
    float fx, fy, fz;
    linear_coef(local_pos, i, j, k, fx, fy, fz);
    int64 bulk_idx = find_bulk(i, j, k);
    if (bulk_idx != -1) {
      int local_i = i % N;
      int local_j = j % N;
      int local_k = k % N;
      float v000 = w_delta(bulk_idx, local_i, local_j, local_k);
      float v001 = w_delta(bulk_idx, local_i + 1, local_j, local_k);
      float v010 = w_delta(bulk_idx, local_i, local_j + 1, local_k);
      float v011 = w_delta(bulk_idx, local_i + 1, local_j + 1, local_k);
      float v100 = w_delta(bulk_idx, local_i, local_j, local_k + 1);
      float v101 = w_delta(bulk_idx, local_i + 1, local_j, local_k + 1);
      float v110 = w_delta(bulk_idx, local_i, local_j + 1, local_k + 1);
      float v111 = w_delta(bulk_idx, local_i + 1, local_j + 1, local_k + 1);
      return trilerp(v000, v001, v010, v011, v100, v101, v110, v111, fx, fy,
                     fz);
    } else {
      return 0;
    }
  }

  float get_omega_x(FLUID::Vec3f &pos) {
    FLUID::Vec3f local_pos =
        pos - FLUID::Vec3f(0.5 * h, 0.0 * h, 0.0 * h) - bmin;
    int i, j, k;
    float fx, fy, fz;
    linear_coef(local_pos, i, j, k, fx, fy, fz);
    int64 bulk_idx = find_bulk(i, j, k);
    if (bulk_idx != -1) {
      int local_i = i % N;
      int local_j = j % N;
      int local_k = k % N;
      float v000 = omega_x(bulk_idx, local_i, local_j, local_k);
      float v001 = omega_x(bulk_idx, local_i + 1, local_j, local_k);
      float v010 = omega_x(bulk_idx, local_i, local_j + 1, local_k);
      float v011 = omega_x(bulk_idx, local_i + 1, local_j + 1, local_k);
      float v100 = omega_x(bulk_idx, local_i, local_j, local_k + 1);
      float v101 = omega_x(bulk_idx, local_i + 1, local_j, local_k + 1);
      float v110 = omega_x(bulk_idx, local_i, local_j + 1, local_k + 1);
      float v111 = omega_x(bulk_idx, local_i + 1, local_j + 1, local_k + 1);
      return trilerp(v000, v001, v010, v011, v100, v101, v110, v111, fx, fy,
                     fz);
    } else {
      return 0;
    }
  }
  float get_omega_y(FLUID::Vec3f &pos) {
    FLUID::Vec3f local_pos =
        pos - FLUID::Vec3f(0.0 * h, 0.5 * h, 0.0 * h) - bmin;
    int i, j, k;
    float fx, fy, fz;
    linear_coef(local_pos, i, j, k, fx, fy, fz);
    int64 bulk_idx = find_bulk(i, j, k);
    if (bulk_idx != -1) {
      int local_i = i % N;
      int local_j = j % N;
      int local_k = k % N;
      float v000 = omega_y(bulk_idx, local_i, local_j, local_k);
      float v001 = omega_y(bulk_idx, local_i + 1, local_j, local_k);
      float v010 = omega_y(bulk_idx, local_i, local_j + 1, local_k);
      float v011 = omega_y(bulk_idx, local_i + 1, local_j + 1, local_k);
      float v100 = omega_y(bulk_idx, local_i, local_j, local_k + 1);
      float v101 = omega_y(bulk_idx, local_i + 1, local_j, local_k + 1);
      float v110 = omega_y(bulk_idx, local_i, local_j + 1, local_k + 1);
      float v111 = omega_y(bulk_idx, local_i + 1, local_j + 1, local_k + 1);
      return trilerp(v000, v001, v010, v011, v100, v101, v110, v111, fx, fy,
                     fz);
    } else {
      return 0;
    }
  }
  float get_omega_z(FLUID::Vec3f &pos) {
    FLUID::Vec3f local_pos =
        pos - FLUID::Vec3f(0.0 * h, 0.0 * h, 0.5 * h) - bmin;
    int i, j, k;
    float fx, fy, fz;
    linear_coef(local_pos, i, j, k, fx, fy, fz);
    int64 bulk_idx = find_bulk(i, j, k);
    if (bulk_idx != -1) {
      int local_i = i % N;
      int local_j = j % N;
      int local_k = k % N;
      float v000 = omega_z(bulk_idx, local_i, local_j, local_k);
      float v001 = omega_z(bulk_idx, local_i + 1, local_j, local_k);
      float v010 = omega_z(bulk_idx, local_i, local_j + 1, local_k);
      float v011 = omega_z(bulk_idx, local_i + 1, local_j + 1, local_k);
      float v100 = omega_z(bulk_idx, local_i, local_j, local_k + 1);
      float v101 = omega_z(bulk_idx, local_i + 1, local_j, local_k + 1);
      float v110 = omega_z(bulk_idx, local_i, local_j + 1, local_k + 1);
      float v111 = omega_z(bulk_idx, local_i + 1, local_j + 1, local_k + 1);
      return trilerp(v000, v001, v010, v011, v100, v101, v110, v111, fx, fy,
                     fz);
    } else {
      return 0;
    }
  }
  float get_d_omega_x(FLUID::Vec3f &pos) {
    FLUID::Vec3f local_pos =
        pos - FLUID::Vec3f(0.5 * h, 0.0 * h, 0.0 * h) - bmin;
    int i, j, k;
    float fx, fy, fz;
    linear_coef(local_pos, i, j, k, fx, fy, fz);
    int64 bulk_idx = find_bulk(i, j, k);
    if (bulk_idx != -1) {
      int local_i = i % N;
      int local_j = j % N;
      int local_k = k % N;
      float v000 = omega_x_delta(bulk_idx, local_i, local_j, local_k);
      float v001 = omega_x_delta(bulk_idx, local_i + 1, local_j, local_k);
      float v010 = omega_x_delta(bulk_idx, local_i, local_j + 1, local_k);
      float v011 = omega_x_delta(bulk_idx, local_i + 1, local_j + 1, local_k);
      float v100 = omega_x_delta(bulk_idx, local_i, local_j, local_k + 1);
      float v101 = omega_x_delta(bulk_idx, local_i + 1, local_j, local_k + 1);
      float v110 = omega_x_delta(bulk_idx, local_i, local_j + 1, local_k + 1);
      float v111 =
          omega_x_delta(bulk_idx, local_i + 1, local_j + 1, local_k + 1);
      return trilerp(v000, v001, v010, v011, v100, v101, v110, v111, fx, fy,
                     fz);
    } else {
      return 0;
    }
  }
  float get_d_omega_y(FLUID::Vec3f &pos) {
    FLUID::Vec3f local_pos =
        pos - FLUID::Vec3f(0.0 * h, 0.5 * h, 0.0 * h) - bmin;
    int i, j, k;
    float fx, fy, fz;
    linear_coef(local_pos, i, j, k, fx, fy, fz);
    int64 bulk_idx = find_bulk(i, j, k);
    if (bulk_idx != -1) {
      int local_i = i % N;
      int local_j = j % N;
      int local_k = k % N;
      float v000 = omega_y_delta(bulk_idx, local_i, local_j, local_k);
      float v001 = omega_y_delta(bulk_idx, local_i + 1, local_j, local_k);
      float v010 = omega_y_delta(bulk_idx, local_i, local_j + 1, local_k);
      float v011 = omega_y_delta(bulk_idx, local_i + 1, local_j + 1, local_k);
      float v100 = omega_y_delta(bulk_idx, local_i, local_j, local_k + 1);
      float v101 = omega_y_delta(bulk_idx, local_i + 1, local_j, local_k + 1);
      float v110 = omega_y_delta(bulk_idx, local_i, local_j + 1, local_k + 1);
      float v111 =
          omega_y_delta(bulk_idx, local_i + 1, local_j + 1, local_k + 1);
      return trilerp(v000, v001, v010, v011, v100, v101, v110, v111, fx, fy,
                     fz);
    } else {
      return 0;
    }
  }
  float get_d_omega_z(FLUID::Vec3f &pos) {
    FLUID::Vec3f local_pos =
        pos - FLUID::Vec3f(0.0 * h, 0.0 * h, 0.5 * h) - bmin;
    int i, j, k;
    float fx, fy, fz;
    linear_coef(local_pos, i, j, k, fx, fy, fz);
    int64 bulk_idx = find_bulk(i, j, k);
    if (bulk_idx != -1) {
      int local_i = i % N;
      int local_j = j % N;
      int local_k = k % N;
      float v000 = omega_z_delta(bulk_idx, local_i, local_j, local_k);
      float v001 = omega_z_delta(bulk_idx, local_i + 1, local_j, local_k);
      float v010 = omega_z_delta(bulk_idx, local_i, local_j + 1, local_k);
      float v011 = omega_z_delta(bulk_idx, local_i + 1, local_j + 1, local_k);
      float v100 = omega_z_delta(bulk_idx, local_i, local_j, local_k + 1);
      float v101 = omega_z_delta(bulk_idx, local_i + 1, local_j, local_k + 1);
      float v110 = omega_z_delta(bulk_idx, local_i, local_j + 1, local_k + 1);
      float v111 =
          omega_z_delta(bulk_idx, local_i + 1, local_j + 1, local_k + 1);
      return trilerp(v000, v001, v010, v011, v100, v101, v110, v111, fx, fy,
                     fz);
    } else {
      return 0;
    }
  }

  float get_s_omega_x(FLUID::Vec3f &pos) {
    FLUID::Vec3f local_pos =
        pos - FLUID::Vec3f(0.5 * h, 0.0 * h, 0.0 * h) - bmin;
    int i, j, k;
    float fx, fy, fz;
    linear_coef(local_pos, i, j, k, fx, fy, fz);
    int64 bulk_idx = find_bulk(i, j, k);
    if (bulk_idx != -1) {
      int local_i = i % N;
      int local_j = j % N;
      int local_k = k % N;
      float v000 = omega_x_save(bulk_idx, local_i, local_j, local_k);
      float v001 = omega_x_save(bulk_idx, local_i + 1, local_j, local_k);
      float v010 = omega_x_save(bulk_idx, local_i, local_j + 1, local_k);
      float v011 = omega_x_save(bulk_idx, local_i + 1, local_j + 1, local_k);
      float v100 = omega_x_save(bulk_idx, local_i, local_j, local_k + 1);
      float v101 = omega_x_save(bulk_idx, local_i + 1, local_j, local_k + 1);
      float v110 = omega_x_save(bulk_idx, local_i, local_j + 1, local_k + 1);
      float v111 =
          omega_x_save(bulk_idx, local_i + 1, local_j + 1, local_k + 1);
      return trilerp(v000, v001, v010, v011, v100, v101, v110, v111, fx, fy,
                     fz);
    } else {
      return 0;
    }
  }
  float get_s_omega_y(FLUID::Vec3f &pos) {
    FLUID::Vec3f local_pos =
        pos - FLUID::Vec3f(0.0 * h, 0.5 * h, 0.0 * h) - bmin;
    int i, j, k;
    float fx, fy, fz;
    linear_coef(local_pos, i, j, k, fx, fy, fz);
    int64 bulk_idx = find_bulk(i, j, k);
    if (bulk_idx != -1) {
      int local_i = i % N;
      int local_j = j % N;
      int local_k = k % N;
      float v000 = omega_y_save(bulk_idx, local_i, local_j, local_k);
      float v001 = omega_y_save(bulk_idx, local_i + 1, local_j, local_k);
      float v010 = omega_y_save(bulk_idx, local_i, local_j + 1, local_k);
      float v011 = omega_y_save(bulk_idx, local_i + 1, local_j + 1, local_k);
      float v100 = omega_y_save(bulk_idx, local_i, local_j, local_k + 1);
      float v101 = omega_y_save(bulk_idx, local_i + 1, local_j, local_k + 1);
      float v110 = omega_y_save(bulk_idx, local_i, local_j + 1, local_k + 1);
      float v111 =
          omega_y_save(bulk_idx, local_i + 1, local_j + 1, local_k + 1);
      return trilerp(v000, v001, v010, v011, v100, v101, v110, v111, fx, fy,
                     fz);
    } else {
      return 0;
    }
  }
  float get_s_omega_z(FLUID::Vec3f &pos) {
    FLUID::Vec3f local_pos =
        pos - FLUID::Vec3f(0.0 * h, 0.0 * h, 0.5 * h) - bmin;
    int i, j, k;
    float fx, fy, fz;
    linear_coef(local_pos, i, j, k, fx, fy, fz);
    int64 bulk_idx = find_bulk(i, j, k);
    if (bulk_idx != -1) {
      int local_i = i % N;
      int local_j = j % N;
      int local_k = k % N;
      float v000 = omega_z_save(bulk_idx, local_i, local_j, local_k);
      float v001 = omega_z_save(bulk_idx, local_i + 1, local_j, local_k);
      float v010 = omega_z_save(bulk_idx, local_i, local_j + 1, local_k);
      float v011 = omega_z_save(bulk_idx, local_i + 1, local_j + 1, local_k);
      float v100 = omega_z_save(bulk_idx, local_i, local_j, local_k + 1);
      float v101 = omega_z_save(bulk_idx, local_i + 1, local_j, local_k + 1);
      float v110 = omega_z_save(bulk_idx, local_i, local_j + 1, local_k + 1);
      float v111 =
          omega_z_save(bulk_idx, local_i + 1, local_j + 1, local_k + 1);
      return trilerp(v000, v001, v010, v011, v100, v101, v110, v111, fx, fy,
                     fz);
    } else {
      return 0;
    }
  }
  float M4_kernel(float rho) {
    if (rho >= 2)
      return 0;
    if (rho >= 1)
      return 0.5 * (2 - rho) * (2 - rho) * (1 - rho);
    return 1 - 2.5 * rho * rho + 1.5 * rho * rho * rho;
  }
  float M4_weight(FLUID::Vec3f &pos, FLUID::Vec3f &grid) {
    float w = 1.0;
    for (int i = 0; i < 3; i++) {
      w *= M4_kernel(fabs((pos[i] - grid[i]) / h));
    }
    return w;
  }
  float get_omega_x_M4(FLUID::Vec3f &pos) {
    FLUID::Vec3f local_pos =
        pos - FLUID::Vec3f(0.5 * h, 0.0 * h, 0.0 * h) - bmin;
    int i, j, k;
    float fx, fy, fz;
    linear_coef(local_pos, i, j, k, fx, fy, fz);
    int64 bulk_idx = find_bulk(i, j, k);
    if (bulk_idx != -1) {
      float sum = 0;
      for (int kk = k - 2; kk <= k + 2; kk++) {
        for (int jj = j - 2; jj <= j + 2; jj++)
          for (int ii = i - 2; ii <= i + 2; ii++) {
            float value00 = omega_x(bulk_idx, ii, jj, kk);
            float weight = M4_weight(local_pos, FLUID::Vec3f(ii, jj, kk) * h);
            sum += value00 * weight;
          }
      }
      return sum;
    } else {
      return 0;
    }
  }
  float get_omega_y_M4(FLUID::Vec3f &pos) {
    FLUID::Vec3f local_pos =
        pos - FLUID::Vec3f(0.0 * h, 0.5 * h, 0.0 * h) - bmin;
    int i, j, k;
    float fx, fy, fz;
    linear_coef(local_pos, i, j, k, fx, fy, fz);
    int64 bulk_idx = find_bulk(i, j, k);
    if (bulk_idx != -1) {
      float sum = 0;
      for (int kk = k - 2; kk <= k + 2; kk++) {
        for (int jj = j - 2; jj <= j + 2; jj++)
          for (int ii = i - 2; ii <= i + 2; ii++) {
            float value00 = omega_y(bulk_idx, ii, jj, kk);
            float weight = M4_weight(local_pos, FLUID::Vec3f(ii, jj, kk) * h);
            sum += value00 * weight;
          }
      }
      return sum;
    } else {
      return 0;
    }
  }
  float get_omega_z_M4(FLUID::Vec3f &pos) {
    FLUID::Vec3f local_pos =
        pos - FLUID::Vec3f(0.0 * h, 0.0 * h, 0.5 * h) - bmin;
    int i, j, k;
    float fx, fy, fz;
    linear_coef(local_pos, i, j, k, fx, fy, fz);
    int64 bulk_idx = find_bulk(i, j, k);
    if (bulk_idx != -1) {
      float sum = 0;
      for (int kk = k - 2; kk <= k + 2; kk++) {
        for (int jj = j - 2; jj <= j + 2; jj++)
          for (int ii = i - 2; ii <= i + 2; ii++) {
            float value00 = omega_z(bulk_idx, ii, jj, kk);
            float weight = M4_weight(local_pos, FLUID::Vec3f(ii, jj, kk) * h);
            sum += value00 * weight;
          }
      }
      return sum;
    } else {
      return 0;
    }
  }
  FLUID::Vec3f get_vorticity_M4(FLUID::Vec3f &pos) {
    return FLUID::Vec3f(get_omega_x_M4(pos), get_omega_y_M4(pos),
                        get_omega_z_M4(pos));
  }
  FLUID::Vec3f get_velocity(FLUID::Vec3f &pos) {
    return FLUID::Vec3f(get_u(pos), get_v(pos), get_w(pos));
  }
  FLUID::Vec3f get_vorticity(FLUID::Vec3f &pos) {
    return FLUID::Vec3f(get_omega_x(pos), get_omega_y(pos), get_omega_z(pos));
  }
  FLUID::Vec3f get_dvorticity(FLUID::Vec3f &pos) {
    return FLUID::Vec3f(get_d_omega_x(pos), get_d_omega_y(pos),
                        get_d_omega_z(pos));
  }
  FLUID::Vec3f get_svorticity(FLUID::Vec3f &pos) {
    return FLUID::Vec3f(get_s_omega_x(pos), get_s_omega_y(pos),
                        get_s_omega_z(pos));
  }
  float get_liquid_phi(FLUID::Vec3f &pos) {
    FLUID::Vec3f local_pos =
        pos - FLUID::Vec3f(0.5 * h, 0.5 * h, 0.5 * h) - bmin;
    int i, j, k;
    float fx, fy, fz;
    linear_coef(local_pos, i, j, k, fx, fy, fz);
    int64 bulk_idx = find_bulk(i, j, k);
    if (bulk_idx != -1) {
      int local_i = i % N;
      int local_j = j % N;
      int local_k = k % N;
      float v000 = liquid_phi(bulk_idx, local_i, local_j, local_k);
      float v001 = liquid_phi(bulk_idx, local_i + 1, local_j, local_k);
      float v010 = liquid_phi(bulk_idx, local_i, local_j + 1, local_k);
      float v011 = liquid_phi(bulk_idx, local_i + 1, local_j + 1, local_k);
      float v100 = liquid_phi(bulk_idx, local_i, local_j, local_k + 1);
      float v101 = liquid_phi(bulk_idx, local_i + 1, local_j, local_k + 1);
      float v110 = liquid_phi(bulk_idx, local_i, local_j + 1, local_k + 1);
      float v111 = liquid_phi(bulk_idx, local_i + 1, local_j + 1, local_k + 1);
      return trilerp(v000, v001, v010, v011, v100, v101, v110, v111, fx, fy,
                     fz);
    } else {
      return 0;
    }
  }
  float get_solid_phi(FLUID::Vec3f &pos) {
    FLUID::Vec3f local_pos = pos - bmin;
    int i, j, k;
    float fx, fy, fz;
    linear_coef(local_pos, i, j, k, fx, fy, fz);
    int64 bulk_idx = find_bulk(i, j, k);
    if (bulk_idx != -1) {
      int local_i = i % N;
      int local_j = j % N;
      int local_k = k % N;
      float v000 = solid_phi(bulk_idx, local_i, local_j, local_k);
      float v001 = solid_phi(bulk_idx, local_i + 1, local_j, local_k);
      float v010 = solid_phi(bulk_idx, local_i, local_j + 1, local_k);
      float v011 = solid_phi(bulk_idx, local_i + 1, local_j + 1, local_k);
      float v100 = solid_phi(bulk_idx, local_i, local_j, local_k + 1);
      float v101 = solid_phi(bulk_idx, local_i + 1, local_j, local_k + 1);
      float v110 = solid_phi(bulk_idx, local_i, local_j + 1, local_k + 1);
      float v111 = solid_phi(bulk_idx, local_i + 1, local_j + 1, local_k + 1);
      return trilerp(v000, v001, v010, v011, v100, v101, v110, v111, fx, fy,
                     fz);
    } else {
      return 0;
    }
  }
  FLUID::Vec3f get_delta_vel(FLUID::Vec3f &pos) {
    return FLUID::Vec3f(get_du(pos), get_dv(pos), get_dw(pos));
  }
  FLUID::Vec3f get_grad_solid(FLUID::Vec3f &pos) {
    FLUID::Vec3f local_pos =
        pos - FLUID::Vec3f(0.5 * h, 0.5 * h, 0.5 * h) - bmin;
    int i, j, k;
    float fx, fy, fz;
    linear_coef(local_pos, i, j, k, fx, fy, fz);
    int64 bulk_idx = find_bulk(i, j, k);
    if (bulk_idx != -1) {
      int local_i = i % N;
      int local_j = j % N;
      int local_k = k % N;
      float v000 = solid_phi(bulk_idx, local_i, local_j, local_k);
      float v100 = solid_phi(bulk_idx, local_i + 1, local_j, local_k);
      float v010 = solid_phi(bulk_idx, local_i, local_j + 1, local_k);
      float v110 = solid_phi(bulk_idx, local_i + 1, local_j + 1, local_k);
      float v001 = solid_phi(bulk_idx, local_i, local_j, local_k + 1);
      float v101 = solid_phi(bulk_idx, local_i + 1, local_j, local_k + 1);
      float v011 = solid_phi(bulk_idx, local_i, local_j + 1, local_k + 1);
      float v111 = solid_phi(bulk_idx, local_i + 1, local_j + 1, local_k + 1);

      float ddx00 = (v100 - v000);
      float ddx10 = (v110 - v010);
      float ddx01 = (v101 - v001);
      float ddx11 = (v111 - v011);
      float dv_dx = bilerp(ddx00, ddx10, ddx01, ddx11, fy, fz);

      float ddy00 = (v010 - v000);
      float ddy10 = (v110 - v100);
      float ddy01 = (v011 - v001);
      float ddy11 = (v111 - v101);
      float dv_dy = bilerp(ddy00, ddy10, ddy01, ddy11, fx, fz);

      float ddz00 = (v001 - v000);
      float ddz10 = (v101 - v100);
      float ddz01 = (v011 - v010);
      float ddz11 = (v111 - v110);
      float dv_dz = bilerp(ddz00, ddz10, ddz01, ddz11, fx, fy);

      return FLUID::Vec3f(dv_dx, dv_dy, dv_dz);
    } else {
      return FLUID::Vec3f(0, 0, 0);
    }
  }
  uint64 get_bulk_index(int i, int j, int k) {
    // int64 idx = i+j*ni+k*ni*nj;
    return index_mapping[FLUID::Vec3i(i, j, k)];
  }
  void SetName(std::string name)
  {
      domainName = name;
  }
  void write_bulk_bgeo(string file_path, int frame)
  {
      std::vector<std::vector<FLIP_particle>> _p;
      std::vector<FLIP_particle> _op;
      _p.resize(fluid_bulk.size());
      for(int i=0;i<fluid_bulk.size();i++)
      {
          _p[i].reserve(512);
      }
      tbb::parallel_for((size_t)0, (size_t)fluid_bulk.size(), (size_t)1, [&](size_t idx) {
          for(int kk=0;kk<8;kk++)for(int jj=0;jj<8;jj++)for(int ii=0;ii<8;ii++) {
                      FLUID::Vec3i ijk = FLUID::Vec3i(fluid_bulk[idx].tile_corner + FLUID::Vec3i(ii,jj,kk));
                      FLUID::Vec3f pos = bmin + FLUID::Vec3f(ijk[0],ijk[1],ijk[2])  * h + FLUID::Vec3f(0.5 * h);
                      FLUID::Vec3f vel = get_velocity(pos);
                      if(get_liquid_phi(pos)<0)
                        _p[idx].emplace_back(FLIP_particle(pos,vel));

                  }
      });
      _op.resize(0);
      for (int j = 0; j < _p.size(); ++j) {
          _op.insert(_op.end(), _p[j].begin(), _p[j].end());
      }
      char file_name[1024];
      sprintf(file_name, "%s_%s_", file_path.c_str(), domainName.c_str(), frame);
      vdbToolsWapper::outputBgeo(file_name, frame, _op);

  }
  void write_bulk_vdb(string file_path, int frame)
  {
      openvdb::GridPtrVec G;
      std::vector<openvdb::GridPtrVec> grids;
      grids.resize(fluid_bulk.size());
      tbb::parallel_for((size_t)0, (size_t)fluid_bulk.size(), (size_t)1, [&](size_t idx){
          openvdb::GridPtrVec g;
          std::vector<FLUID::Vec3i> voxels;
          std::vector<FLUID::Vec3f> vels;
          for(int i=0;i<n_perbulk;i++)
          {
              FLUID::Vec3i ijk(i%8, (i/8)%8, i/64);
              voxels.emplace_back(fluid_bulk[idx].tile_corner*8 + ijk);
              float uu = 0.5f*(u(idx, ijk[0],ijk[1],ijk[2]) + u(idx, ijk[0]+1, ijk[1], ijk[2]));
              float vv = 0.5f*(v(idx, ijk[0],ijk[1],ijk[2]) + v(idx, ijk[0], ijk[1]+1, ijk[2]));
              float ww = 0.5f*(w(idx, ijk[0],ijk[1],ijk[2]) + w(idx, ijk[0], ijk[1], ijk[2]+1));
              vels.emplace_back(FLUID::Vec3f(uu,vv,ww));
          }
          FLUID::Vec3f offset = bmin + h*FLUID::Vec3f(0.5);
          vdbToolsWapper::genVDBVelGrid(voxels, vels, h, offset, g);
          grids[idx] = g;
      });
      for(auto g:grids)
      {
          for(auto gg:g)
          {
              G.push_back(gg);
          }
      }
      char file_name[256];
      sprintf(file_name, "%s_%s_%04d.vdb", file_path.c_str(), domainName.c_str(), frame);
      std::string vdbname(file_name);
      openvdb::io::File file(vdbname);
      file.write(G);
      file.close();
  }
  void write_bulk_obj(string file_path, int frame) {
    cout << "ouput bulks" << endl;
    ostringstream strout;
    char zero = '0';
    strout << file_path << "_bulk_" <<domainName<< frame << ".obj";

    string filename = strout.str();
    ofstream outfile(filename.c_str());

    for (unsigned int i = 0; i < n_bulks; i++) {
      FLUID::Vec3f c_pos = FLUID::Vec3f(fluid_bulk[i].tile_corner) * h + bmin;
      outfile << "v"
              << " " << c_pos[0] << " " << c_pos[1] << " " << c_pos[2] << endl;
      outfile << "v"
              << " " << c_pos[0] + bulk_size << " " << c_pos[1] << " "
              << c_pos[2] << endl;
      outfile << "v"
              << " " << c_pos[0] + bulk_size << " " << c_pos[1] + bulk_size
              << " " << c_pos[2] << endl;
      outfile << "v"
              << " " << c_pos[0] << " " << c_pos[1] + bulk_size << " "
              << c_pos[2] << endl;

      outfile << "v"
              << " " << c_pos[0] << " " << c_pos[1] << " "
              << c_pos[2] + bulk_size << endl;
      outfile << "v"
              << " " << c_pos[0] + bulk_size << " " << c_pos[1] << " "
              << c_pos[2] + bulk_size << endl;
      outfile << "v"
              << " " << c_pos[0] + bulk_size << " " << c_pos[1] + bulk_size
              << " " << c_pos[2] + bulk_size << endl;
      outfile << "v"
              << " " << c_pos[0] << " " << c_pos[1] + bulk_size << " "
              << c_pos[2] + bulk_size << endl;
    }
    for (unsigned int i = 0; i < n_bulks; i++) {
      uint off_set = i * 8;
      outfile << "f"
              << " " << 4 + off_set << " " << 3 + off_set << " " << 2 + off_set
              << " " << 1 + off_set << endl;
      outfile << "f"
              << " " << 8 + off_set << " " << 7 + off_set << " " << 6 + off_set
              << " " << 5 + off_set << endl;
      outfile << "f"
              << " " << 4 + off_set << " " << 3 + off_set << " " << 7 + off_set
              << " " << 8 + off_set << endl;
      outfile << "f"
              << " " << 1 + off_set << " " << 2 + off_set << " " << 6 + off_set
              << " " << 5 + off_set << endl;
    }
  }
};
typedef sparse_fluid_3D<8> sparse_fluid8x8x8;
typedef sparse_fluid_3D<6> sparse_fluid6x6x6;

#endif
