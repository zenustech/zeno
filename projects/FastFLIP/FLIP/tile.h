#ifndef TILE_H
#define TILE_H

#include "vec.h"
#include <cassert>
#include <vector>
#include <atomic>
using namespace std;

template <typename T, int N> struct chunck3D {
  vector<T> data;
  chunck3D(void) {
    data.resize(N * N * N);
    data.assign(N * N * N, (T)0);
  }
  ~chunck3D() {
    data.resize(0);
    data.shrink_to_fit();
  }

  T &operator()(int i, int j, int k) {
    assert(i >= 0 && i < N);
    assert(j >= 0 && j < N);
    assert(k >= 0 && k < N);
    return data[i + N * (j + N * k)];
  }

  T const &operator()(int i, int j, int k) const {
    assert(i >= 0 && i < N);
    assert(j >= 0 && j < N);
    assert(k >= 0 && k < N);
    return data[i + N * (j + N * k)];
  }
};

template <int N> struct fluid_Tile {
  bool mark, mask;

  chunck3D<float, N> u;
  chunck3D<float, N> u_solid;
  chunck3D<float, N> u_save;
  chunck3D<float, N> u_delta;
  chunck3D<float, N> u_coef;
  chunck3D<float, N> u_extrapolate;
  chunck3D<float, N> v;
  chunck3D<float, N> v_solid;
  chunck3D<float, N> v_save;
  chunck3D<float, N> v_delta;
  chunck3D<float, N> v_coef;
  chunck3D<float, N> v_extrapolate;
  chunck3D<float, N> w;
  chunck3D<float, N> w_solid;
  chunck3D<float, N> w_save;
  chunck3D<float, N> w_delta;
  chunck3D<float, N> w_coef;
  chunck3D<float, N> w_extrapolate;
  chunck3D<float, N> liquid_phi;
  chunck3D<float, N> solid_phi;
  chunck3D<float, N> u_weight;
  chunck3D<float, N> v_weight;
  chunck3D<float, N> w_weight;

  chunck3D<float, N> omega_x;
  chunck3D<float, N> omega_y;
  chunck3D<float, N> omega_z;
  chunck3D<float, N> omega_x_save;
  chunck3D<float, N> omega_y_save;
  chunck3D<float, N> omega_z_save;
  chunck3D<float, N> omega_x_delta;
  chunck3D<float, N> omega_y_delta;
  chunck3D<float, N> omega_z_delta;

  chunck3D<float, N> psi_x;
  chunck3D<float, N> psi_y;
  chunck3D<float, N> psi_z;

  chunck3D<char, N> u_valid;
  chunck3D<char, N> v_valid;
  chunck3D<char, N> w_valid;
  chunck3D<char, N> old_valid;
  chunck3D<double, N> pressure;
  chunck3D<uint, N> global_index;
  //std::atomic_uchar cnt[N*N*N];

  FLUID::Vec3i tile_corner;
  uint bulk_index;
  bool is_boundary;
  bool has_hole=false;
  // INT64 right_bulk; //i+1
  // INT64 left_bulk;  //i-1
  // INT64 top_bulk;   //j+1
  // INT64 bottom_bulk;//j-1
  // INT64 front_bulk; //k-1
  // INT64 back_bulk;  //k+1
  fluid_Tile() {}
  fluid_Tile(const FLUID::Vec3i &corner, uint index) {
    tile_corner = corner;
    bulk_index = index;
    is_boundary = false;
    // right_bulk = -1;
    // left_bulk  = -1;
    // top_bulk   = -1;
    // bottom_bulk= -1;
    // front_bulk = -1;
    // back_bulk  = -1;
  }
  ~fluid_Tile() {}
};

#endif