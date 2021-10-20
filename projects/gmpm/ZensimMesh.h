#pragma once
#include "zensim/math/Vec.h"
#include "zensim/math/matrix/Matrix.hpp"
#include <fstream>
#include <iostream>
#include <zeno/types/PrimitiveObject.h>

namespace zeno {

struct ZenoFEMMesh : zeno::IObject {

  using value_type = float;
  using size_type = std::size_t;
  using spmat_type = zs::YaleSparseMatrix<value_type, int>;
  using allocator_type = zs::ZSPmrAllocator<>;

  using vec3 = zs::vec<value_type, 3>;
  using vec3i = zs::vec<int, 3>;
  using vec4i = zs::vec<int, 4>;
  using mat3 = zs::vec<value_type, 3, 3>;
  using mat4 = zs::vec<value_type, 4, 4>;
  using mat_9_12 = zs::vec<value_type, 9, 12>;

  std::shared_ptr<zeno::PrimitiveObject> _mesh;

  /// counterparts to primitive object
  zs::Vector<vec3> _X, _V;
  zs::Vector<vec4i> _tets;
  zs::Vector<vec3i> _tris;

  zs::Vector<int> _bouDoFs;
  zs::Vector<int> _freeDoFs;
  zs::Vector<int> _DoF2FreeDoF;

  zs::Vector<value_type> _elmMass;
  zs::Vector<value_type> _elmVolume;
  zs::Vector<mat_9_12> _elmdFdx;
  zs::Vector<mat4> _elmMinv;
  zs::Vector<mat3> _elmDmInv;

  zs::Vector<value_type> _elmYoungModulus;
  zs::Vector<value_type> _elmPoissonRatio;
  zs::Vector<value_type> _elmDensity;

  // spmat_type _connMatrix;
  // spmat_type _freeConnMatrix;
  // zs::Vector<int> _SpMatFreeDoFs;
  // zs::Vector<zs::vec<int, 12, 12>> _elmSpIndices;

  zs::Vector<mat3> _elmAct;
  zs::Vector<mat3> _elmOrient;
  zs::Vector<vec3> _elmWeight;

  zs::Vector<std::size_t> _closeBindPoints;
  zs::Vector<std::size_t> _farBindPoints;

  decltype(auto) get_default_allocator(zs::memsrc_e mre,
                                       zs::ProcID devid) const {
    using namespace zs;
    if constexpr (is_virtual_zs_allocator<allocator_type>::value)
      return get_virtual_memory_source(
          mre, devid, (std::size_t)1 << (std::size_t)36, "STACK");
    else
      return get_memory_source(mre, devid);
  }

  ZenoFEMMesh(const allocator_type &allocator)
      : _X{allocator, 0}, _tets{allocator, 0}, _tris{allocator, 0},
        _bouDoFs{allocator, 0}, _freeDoFs{allocator, 0}, _DoF2FreeDoF{allocator,
                                                                      0},
        _elmMass{allocator, 0}, _elmVolume{allocator, 0},
        _elmdFdx{allocator, 0}, _elmMinv{allocator, 0}, _elmDmInv{allocator, 0},
        _elmYoungModulus{allocator, 0}, _elmPoissonRatio{allocator, 0},
        _elmDensity{allocator, 0}, _elmAct{allocator, 0},
        _elmOrient{allocator, 0}, _elmWeight{allocator, 0}, _closeBindPoints{allocator, 0}, _farBindPoints{allocator, 0} {}

  ZenoFEMMesh(zs::memsrc_e mre = zs::memsrc_e::host, zs::ProcID devid = -1)
      : ZenoFEMMesh{get_default_allocator(mre, devid)} {}

  void relocate(zs::memsrc_e mre = zs::memsrc_e::host, zs::ProcID devid = -1) {
    if (_bouDoFs.memspace() == mre && _bouDoFs.devid() == devid)
      return;
    auto newAllocator = _bouDoFs.get_default_allocator(mre, devid);

    _X = _X.clone(newAllocator);
    _tets = _tets.clone(newAllocator);
    _tris = _tris.clone(newAllocator);
    _bouDoFs = _bouDoFs.clone(newAllocator);
    _freeDoFs = _freeDoFs.clone(newAllocator);
    _DoF2FreeDoF = _DoF2FreeDoF.clone(newAllocator);
    _elmMass = _elmMass.clone(newAllocator);
    _elmVolume = _elmVolume.clone(newAllocator);
    _elmdFdx = _elmdFdx.clone(newAllocator);
    _elmMinv = _elmMinv.clone(newAllocator);
    _elmDmInv = _elmDmInv.clone(newAllocator);
    _elmYoungModulus = _elmYoungModulus.clone(newAllocator);
    _elmPoissonRatio = _elmPoissonRatio.clone(newAllocator);
    _elmDensity = _elmDensity.clone(newAllocator);
    _elmAct = _elmAct.clone(newAllocator);
    _elmOrient = _elmOrient.clone(newAllocator);
    _elmWeight = _elmWeight.clone(newAllocator);
  }

  void DoPreComputation() {
    auto nm_elms = _mesh->quads.size(); // num tets
    for (std::size_t elm_id = 0; elm_id != nm_elms; ++elm_id) {
      auto elm = _mesh->quads[elm_id];
      mat4 M{};
      for (std::size_t i = 0; i != 4; ++i) {
        const auto &vert = _mesh->verts[elm[i]];
        for (std::size_t d = 0; d != 3; ++d)
          M(d, i) = vert[d];
        M(3, i) = 1;
        // M.block(0, i, 3, 1) << vert[0], vert[1], vert[2];
      }
      // M.bottomRows(1).setConstant(1.0);
      _elmVolume[elm_id] = gcem::abs(zs::determinant(M)) / 6;
      _elmMass[elm_id] = _elmVolume[elm_id] * _elmDensity[elm_id];

      mat3 Dm{};
      for (size_t i = 1; i < 4; ++i) {
        const auto &vert = _mesh->verts[elm[i]];
        const auto &vert0 = _mesh->verts[elm[0]];

        for (std::size_t d = 0; d != 3; ++d)
          Dm(d, i - 1) = vert[d] - vert0[d];
        // Dm.col(i - 1) << vert[0] - vert0[0], vert[1] - vert0[1],
        //    vert[2] - vert0[2];
      }

      mat3 DmInv = inverse(Dm);
      _elmMinv[elm_id] = inverse(M);
      _elmDmInv[elm_id] = DmInv;

      value_type m = DmInv(0, 0);
      value_type n = DmInv(0, 1);
      value_type o = DmInv(0, 2);
      value_type p = DmInv(1, 0);
      value_type q = DmInv(1, 1);
      value_type r = DmInv(1, 2);
      value_type s = DmInv(2, 0);
      value_type t = DmInv(2, 1);
      value_type u = DmInv(2, 2);

      value_type t1 = -m - p - s;
      value_type t2 = -n - q - t;
      value_type t3 = -o - r - u;

      _elmdFdx[elm_id] =
          mat_9_12{t1, 0, 0, m, 0, 0, p, 0, 0,  s, 0, 0, 0, t1, 0, 0, m, 0,
                   0,  p, 0, 0, s, 0, 0, 0, t1, 0, 0, m, 0, 0,  p, 0, 0, s,
                   t2, 0, 0, n, 0, 0, q, 0, 0,  t, 0, 0, 0, t2, 0, 0, n, 0,
                   0,  q, 0, 0, t, 0, 0, 0, t2, 0, 0, n, 0, 0,  q, 0, 0, t,
                   t3, 0, 0, o, 0, 0, r, 0, 0,  u, 0, 0, 0, t3, 0, 0, o, 0,
                   0,  r, 0, 0, u, 0, 0, 0, t3, 0, 0, o, 0, 0,  r, 0, 0, u};
    }
  }
  // load .node file -> _mesh->attr("pos")
  void LoadVerticesFromFile(const std::string &filename) {
    size_t num_vertices, space_dimension, d1, d2;
    std::ifstream node_fin;
    try {
      node_fin.open(filename.c_str());
      if (!node_fin.is_open())
        std::cerr << "ERROR::NODE::FAILED::" << filename << std::endl;

      node_fin >> num_vertices >> space_dimension >> d1 >> d2;
      auto &pos = _mesh->add_attr<vec3f>("pos");
      pos.resize(num_vertices);

      for (size_t vert_id = 0; vert_id != num_vertices; ++vert_id) {
        node_fin >> d1;
        for (size_t i = 0; i != space_dimension; ++i)
          node_fin >> pos[vert_id][i];
      }
      node_fin.close();

      _X.resize(num_vertices);
#if 0
      for (auto &&[dst, src] : zs::zip(_X, pos))
        dst = vec3{src[0], src[1], src[2]};
#else
      memcpy(_X.data(), pos.data(), sizeof(vec3) * num_vertices);
#endif

    } catch (std::exception &e) {
      std::cerr << e.what() << std::endl;
    }
  }

  // load .ele file -> _mesh->quads
  void LoadElementsFromFile(const std::string &filename) {
    size_t nm_elms, elm_size, v_start_idx, elm_idx;
    std::ifstream ele_fin;
    try {
      ele_fin.open(filename.c_str());
      if (!ele_fin.is_open())
        std::cerr << "ERROR::TET::FAILED::" << filename << std::endl;

      ele_fin >> nm_elms >> elm_size >> v_start_idx;

      auto &quads = _mesh->quads;
      quads.resize(nm_elms);

      for (size_t elm_id = 0; elm_id < nm_elms; ++elm_id) {
        ele_fin >> elm_idx;
        for (size_t i = 0; i < elm_size; ++i) {
          ele_fin >> quads[elm_id][i];
          quads[elm_id][i] -= v_start_idx;
        }
      }
      ele_fin.close();

      _tets.resize(nm_elms);
#if 0
      for (auto &&[dst, src] : zs::zip(_tets, quads))
        dst = vec4i{src[0], src[1], src[2], src[3]};
#else
      memcpy(_tets.data(), quads.data(), sizeof(vec4i) * nm_elms);
#endif

      for (size_t i = 0; i != nm_elms; ++i) {
        auto tet = _mesh->quads[i];
        _mesh->tris.emplace_back(tet[0], tet[1], tet[2]);
        _mesh->tris.emplace_back(tet[1], tet[3], tet[2]);
        _mesh->tris.emplace_back(tet[0], tet[2], tet[3]);
        _mesh->tris.emplace_back(tet[0], tet[3], tet[1]);
      }

      _tris.resize(_mesh->tris.size());
#if 0
      for (auto &&[dst, src] : zs::zip(_tris, _mesh->tris))
        dst = vec3i{src[0], src[1], src[2]};
#else
      memcpy(_tris.data(), _mesh->tris.data(),
             sizeof(typename ZenoFEMMesh::vec3i) * _tris.size());
#endif

    } catch (std::exception &e) {
      std::cerr << e.what() << std::endl;
    }
  }

  // constrained dof indices -> _bouDoFs
  void LoadBoundaryIndicesFromFile(const std::string &filename) {
    size_t nm_cons_dofs, start_idx;
    std::ifstream bou_fin;
    try {
      bou_fin.open(filename.c_str());
      if (!bou_fin.is_open())
        std::cerr << "ERROR::BOU::FAILED::" << filename << std::endl;

      bou_fin >> nm_cons_dofs >> start_idx;
      _bouDoFs.resize(nm_cons_dofs);
      for (size_t c_id = 0; c_id < nm_cons_dofs; ++c_id) {
        bou_fin >> _bouDoFs[c_id];
        _bouDoFs[c_id] -= start_idx;
      }
      bou_fin.close();
    } catch (std::exception &e) {
      std::cerr << e.what() << std::endl;
    }
  }

  // unconstrained dof indices -> _freeDoFs
  void UpdateDoFsMapping() {
    size_t nmDoFs = _mesh->verts.size() * 3;
    size_t nmBouDoFs = _bouDoFs.size();
    size_t nmFreeDoFs = nmDoFs - _bouDoFs.size();
    _freeDoFs.resize(nmFreeDoFs);

    for (size_t cdof_idx = 0, dof = 0, ucdof_count = 0; dof != nmDoFs; ++dof) {
      if (cdof_idx >= nmBouDoFs || dof != _bouDoFs[cdof_idx])
        _freeDoFs[ucdof_count++] = dof;
      else
        ++cdof_idx;
    }
    // build uc mapping
    //_DoF2FreeDoF.resize(nmDoFs, -1);
    std::vector<int> DoF2FreeDoF(nmDoFs, -1);
    for (size_t i = 0; i != nmFreeDoFs; ++i) {
      int ucdof = _freeDoFs[i];
      DoF2FreeDoF[ucdof] = i;
    }
    _DoF2FreeDoF.resize(nmDoFs);
    memcpy(_DoF2FreeDoF.data(), DoF2FreeDoF.data(), sizeof(int) * nmDoFs);

#if 0
    // Initialize connectivity matrices
    size_t nm_elms = _mesh->quads.size();
    std::set<Triplet, triplet_cmp> connTriplets;
    size_t nm_insertions = 0;
    for (size_t elm_id = 0; elm_id < nm_elms; ++elm_id) {
      const auto &elm = _mesh->quads[elm_id];
      for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
          for (size_t k = 0; k < 3; ++k)
            for (size_t l = 0; l < 3; ++l) {
              size_t row = elm[i] * 3 + k;
              size_t col = elm[j] * 3 + l;
              if (row > col)
                continue;
              if (row == col) {
                connTriplets.insert(Triplet(row, col, 1.0));
                ++nm_insertions;
              } else {
                connTriplets.insert(Triplet(row, col, 1.0));
                connTriplets.insert(Triplet(col, row, 1.0));
                nm_insertions += 2;
              }
            }
    }
    _connMatrix = SpMat(nmDoFs, nmDoFs);
    _connMatrix.setFromTriplets(connTriplets.begin(), connTriplets.end());
    _connMatrix.makeCompressed();

    std::set<Triplet, triplet_cmp> freeConnTriplets;
    nm_insertions = 0;
    for (size_t elm_id = 0; elm_id < nm_elms; ++elm_id) {
      const auto &elm = _mesh->quads[elm_id];
      for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
          for (size_t k = 0; k < 3; ++k)
            for (size_t l = 0; l < 3; ++l) {
              size_t row = _DoF2FreeDoF[elm[i] * 3 + k];
              size_t col = _DoF2FreeDoF[elm[j] * 3 + l];
              if (row == -1 || col == -1 || row > col)
                continue;
              if (row == col) {
                freeConnTriplets.insert(Triplet(row, col, 1.0));
                nm_insertions++;
              } else {
                freeConnTriplets.insert(Triplet(row, col, 1.0));
                freeConnTriplets.insert(Triplet(col, row, 1.0));
                nm_insertions += 2;
              }
            }
    }
    _freeConnMatrix = SpMat(nmFreeDoFs, nmFreeDoFs);
    _freeConnMatrix.setFromTriplets(freeConnTriplets.begin(),
                                    freeConnTriplets.end());
    _freeConnMatrix.makeCompressed();

    _SpMatFreeDoFs.resize(_freeConnMatrix.nonZeros());
    size_t uc_idx = 0;
    size_t idx = 0;
    for (size_t k = 0; k != size_t(_connMatrix.outerSize()); ++k)
      for (SpMat::InnerIterator it(_connMatrix, k); it; ++it) {
        size_t row = it.row();
        size_t col = it.col();
        if (_DoF2FreeDoF[row] == -1 || _DoF2FreeDoF[col] == -1) {
          idx++;
          continue;
        }
        _SpMatFreeDoFs[uc_idx] = idx;
        ++uc_idx;
        ++idx;
      }
#endif
  }

};

} // namespace zeno