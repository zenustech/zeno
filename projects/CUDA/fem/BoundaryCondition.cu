#include "../Structures.hpp"
#include "../Utils.hpp"
#include "zensim/Logger.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/types/Property.h"
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>

namespace zeno {

struct ApplyBoundaryOnVertices : INode {
  template <typename LsView, typename TileVecT>
  constexpr void projectBoundary(zs::CudaExecutionPolicy &cudaPol, LsView lsv,
                                 const ZenoBoundary &boundary, TileVecT &verts,
                                 float dt) {
    using namespace zs;
    auto collider = boundary.getBoundary(lsv);
    cudaPol(Collapse{verts.size()},
            [verts = proxy<execspace_e::cuda>({}, verts), boundary = collider,
             dt] __device__(int vi) mutable {
              using mat3 = zs::vec<double, 3, 3>;
              auto vel = verts.template pack<3>("v", vi);
              auto pos = verts.template pack<3>("x", vi);
              if (boundary.queryInside(pos)) {
                auto v = boundary.getVelocity(pos);
                auto n = boundary.getNormal(pos);
                if (v.l2NormSqr() == 0)
                  verts("BCfixed", vi) = 1;
                if (boundary.type == collider_e::Sticky) {
                  verts.template tuple<9>("BCbasis", vi) = mat3::identity();
                  verts("BCorder", vi) = 3;
                  verts.template tuple<3>("BCtarget", vi) =
                      verts.template pack<3>("BCtarget", vi) + dt * v;
                } else if (boundary.type == collider_e::Slip) {
                  auto BCbasis = verts.template pack<3, 3>("BCbasis", vi);
                  int BCorder = verts("BCorder", vi);
                  if (BCorder >= 3)
                    return;
                  for (int d = 0; d != BCorder; ++d) {
                    auto nd = col(BCbasis, d);
                    // remove components in previous normal directions
                    n -= n.dot(nd) * nd;
                  }
                  if (n.l2NormSqr() > limits<float>::epsilon()) {
                    n = n.normalized();
                    for (int d = 0; d != 3; ++d)
                      BCbasis(d, BCorder) = n(d);
                    verts.template tuple<9>("BCbasis", vi) = BCbasis;
                    verts("BCorder", vi) = (BCorder + 1);
                    verts.template tuple<3>("BCtarget", vi) =
                        verts.template pack<3>("BCtarget", vi) +
                        dt * v.dot(n) * n;
                  }
                }
              }
            });
  }
  void apply() override {
    auto zsverts = get_input<ZenoParticles>("ZSParticles");
    auto &verts = zsverts->getParticles<true>();
    auto dt = get_input2<float>("dt");

    using namespace zs;

    auto cudaPol = cuda_exec().device(0);

    /// init BC status
    cudaPol(Collapse{verts.size()},
            [verts = proxy<execspace_e::cuda>({}, verts)] __device__(
                int vi) mutable {
              using mat3 = zs::vec<zs::f64, 3, 3>;
              verts.tuple<9>("BCbasis", vi) = mat3::identity();
              verts("BCorder", vi) = 0;
              verts("BCfixed", vi) = 0;
              verts.tuple<3>("BCtarget", vi) = verts.pack<3>("x", vi);
            });

    if (has_input<ZenoBoundary>("ZSBoundary")) {
      auto boundary = get_input<ZenoBoundary>("ZSBoundary");

      using basic_ls_t = typename ZenoLevelSet::basic_ls_t;
      using const_sdf_vel_ls_t = typename ZenoLevelSet::const_sdf_vel_ls_t;
      using const_transition_ls_t =
          typename ZenoLevelSet::const_transition_ls_t;
      if (boundary->zsls)
        match([&](const auto &ls) {
          if constexpr (is_same_v<RM_CVREF_T(ls), basic_ls_t>) {
            match([&](const auto &lsPtr) {
              auto lsv = get_level_set_view<execspace_e::cuda>(lsPtr);
              projectBoundary(cudaPol, lsv, *boundary, verts, dt);
            })(ls._ls);
          } else if constexpr (is_same_v<RM_CVREF_T(ls), const_sdf_vel_ls_t>) {
            match([&](auto lsv) {
              projectBoundary(cudaPol, SdfVelFieldView{lsv}, *boundary, verts,
                              dt);
            })(ls.template getView<execspace_e::cuda>());
          } else if constexpr (is_same_v<RM_CVREF_T(ls),
                                         const_transition_ls_t>) {
            match([&](auto fieldPair) {
              auto &fvSrc = std::get<0>(fieldPair);
              auto &fvDst = std::get<1>(fieldPair);
              projectBoundary(cudaPol,
                              TransitionLevelSetView{SdfVelFieldView{fvSrc},
                                                     SdfVelFieldView{fvDst},
                                                     ls._stepDt, ls._alpha},
                              *boundary, verts, dt);
            })(ls.template getView<zs::execspace_e::cuda>());
          }
        })(boundary->zsls->getLevelSet());

      if (boundary->type != collider_e::Sticky) {
        cudaPol(Collapse{verts.size()},
                [verts = proxy<execspace_e::cuda>({}, verts)] __device__(
                    int vi) mutable {
                  using mat3 = zs::vec<float, 3, 3>;
                  auto BCbasis = verts.pack<3, 3>("BCbasis", vi);
                  int BCorder = verts("BCorder", vi);
                  if (BCorder != 0) {
                    if (BCorder == 1) {
                      auto n0 = col(BCbasis, 0);
                      auto n1 = n0.orthogonal().normalized();
                      auto n2 = n0.cross(n1).normalized();
                      for (int d = 0; d != 3; ++d) {
                        BCbasis(d, 1) = n1(d);
                        BCbasis(d, 2) = n2(d);
                      }
                    } else if (BCorder == 2) {
                      auto n0 = col(BCbasis, 0);
                      auto n1 = col(BCbasis, 1);
                      auto n2 = n0.cross(n1).normalized();
                      for (int d = 0; d != 3; ++d)
                        BCbasis(d, 2) = n2(d);
                    }
                    verts.tuple<9>("BCbasis", vi) = BCbasis;
                    verts.tuple<3>("BCtarget", vi) =
                        BCbasis.transpose() * verts.pack<3>("BCtarget", vi);
                  }
                });
      }
    }

    set_output("ZSParticles", zsverts);
  }
};

ZENDEFNODE(ApplyBoundaryOnVertices,
           {
               {"ZSParticles", "ZSBoundary", {"float", "dt", "0.01"}},
               {"ZSParticles"},
               {},
               {"FEM"},
           });

struct BindVerticesOnBoundary : INode {
  using tiles_t = typename ZenoParticles::particles_t;
  using dtiles_t = typename ZenoParticles::dtiles_t;
  using T = typename tiles_t::value_type;
  using Ti = int;
  static_assert(sizeof(Ti) == sizeof(T) && std::is_signed_v<Ti>,
                "T and Ti should have the same size");
  using TV = zs::vec<T, 3>;
  using IV = zs::vec<Ti, 3>;
  using bvh_t = zs::LBvh<3, 32, int, T>;
  using bv_t = zs::AABBBox<3, T>;

  zs::Vector<bv_t> retrieve_bounding_volumes(zs::CudaExecutionPolicy &pol,
                                             const tiles_t &vtemp,
                                             const zs::SmallString &xTag,
                                             const tiles_t &eles) {
    using namespace zs;
    using bv_t = AABBBox<3, T>;
    constexpr auto space = execspace_e::cuda;
    zs::Vector<bv_t> ret{eles.get_allocator(), eles.size()};
    pol(range(eles.size()), [eles = proxy<space>({}, eles),
                             bvs = proxy<space>(ret),
                             vtemp = proxy<space>({}, vtemp),
                             xTag] ZS_LAMBDA(int ei) mutable {
      constexpr int dim = 3;
      auto inds =
          eles.template pack<dim>("inds", ei).template reinterpret_bits<int>();
      auto x0 = vtemp.template pack<3>(xTag, inds[0]);
      bv_t bv{x0, x0};
      for (int d = 1; d != dim; ++d)
        merge(bv, vtemp.template pack<3>(xTag, inds[d]));
      bvs[ei] = bv;
    });
    return ret;
  }

  static constexpr T distance(const bv_t &bv, const TV &x) {
    const auto &[mi, ma] = bv;
    TV center = (mi + ma) / 2;
    TV point = (x - center).abs() - (ma - mi) / 2;
    T max = zs::limits<T>::lowest();
    for (int d = 0; d != 3; ++d) {
      if (point[d] > max)
        max = point[d];
      if (point[d] < 0)
        point[d] = 0;
    }
    return (max < 0 ? max : (T)0) + point.norm();
  }

  // ref: https://www.geometrictools.com/GTE/Mathematics/DistPointTriangle.h
  static constexpr auto dist_pt_sqr(const TV &p, const TV &t0, const TV &t1,
                                    const TV &t2, TV &ws) noexcept {
    TV diff = t0 - p;
    TV e0 = t1 - t0;
    TV e1 = t2 - t0;
    T a00 = dot(e0, e0);
    T a01 = dot(e0, e1);
    T a11 = dot(e1, e1);
    T b0 = dot(diff, e0);
    T b1 = dot(diff, e1);
    T det = std::max(a00 * a11 - a01 * a01, (T)0);
    T s = a01 * b1 - a11 * b0;
    T t = a01 * b0 - a00 * b1;

    if (s + t <= det) {
      if (s < (T)0) {
        if (t < (T)0) { // region 4
          if (b0 < (T)0) {
            t = (T)0;
            if (-b0 >= a00)
              s = (T)1;
            else
              s = -b0 / a00;
          } else {
            s = (T)0;
            if (b1 >= (T)0)
              t = (T)0;
            else if (-b1 >= a11)
              t = (T)1;
            else
              t = -b1 / a11;
          }
        } else { // region 3
          s = (T)0;
          if (b1 >= (T)0)
            t = (T)0;
          else if (-b1 >= a11)
            t = (T)1;
          else
            t = -b1 / a11;
        }
      } else if (t < (T)0) { // region 5
        t = (T)0;
        if (b0 >= (T)0)
          s = (T)0;
        else if (-b0 >= a00)
          s = (T)1;
        else
          s = -b0 / a00;
      } else { // region 0
               // minimum at interior point
        s /= det;
        t /= det;
      }
    } else {
      T tmp0{}, tmp1{}, numer{}, denom{};
      if (s < (T)0) { // region 2
        tmp0 = a01 + b0;
        tmp1 = a11 + b1;
        if (tmp1 > tmp0) {
          numer = tmp1 - tmp0;
          denom = a00 - (a01 + a01) + a11;
          if (numer >= denom) {
            s = (T)1;
            t = (T)0;
          } else {
            s = numer / denom;
            t = (T)1 - s;
          }
        } else {
          s = (T)0;
          if (tmp1 <= (T)0)
            t = (T)1;
          else if (b1 >= (T)0)
            t = (T)0;
          else
            t = -b1 / a11;
        }
      } else if (t < (T)0) { // region 6
        tmp0 = a01 + b1;
        tmp1 = a00 + b0;
        if (tmp1 > tmp0) {
          numer = tmp1 - tmp0;
          denom = a00 - (a01 + a01) + a11;
          if (numer >= denom) {
            t = (T)1;
            s = (T)0;
          } else {
            t = numer / denom;
            s = (T)1 - t;
          }
        } else {
          t = (T)0;
          if (tmp1 <= (T)0)
            s = (T)1;
          else if (b0 >= (T)0)
            s = (T)0;
          else
            s = -b0 / a00;
        }
      } else { // region 1
        numer = a11 + b1 - a01 - b0;
        if (numer <= (T)0) {
          s = (T)0;
          t = (T)1;
        } else {
          denom = a00 - (a01 + a01) + a11;
          if (numer >= denom) {
            s = (T)1;
            t = (T)0;
          } else {
            s = numer / denom;
            t = (T)1 - s;
          }
        }
      }
    }
    auto hitpoint = t0 + s * e0 + t * e1;
    ws[0] = 1 - s - t;
    ws[1] = s;
    ws[2] = t;
    return (p - hitpoint).l2NormSqr();
  }
  static constexpr auto dist_pt(const TV &p, const TV &t0, const TV &t1,
                                const TV &t2, TV &ws) {
    return zs::sqrt(dist_pt_sqr(p, t0, t1, t2, ws));
  }

  template <typename LsView>
  void bindBoundary(zs::CudaExecutionPolicy &cudaPol, LsView lsv,
                    dtiles_t &verts, const bvh_t &bvh, const tiles_t &bouverts,
                    const tiles_t &boueles, T distCap) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    cudaPol(Collapse{verts.size()}, [verts = proxy<space>({}, verts),
                                     bouverts = proxy<space>({}, bouverts),
                                     boueles = proxy<space>({}, boueles),
                                     bvh = proxy<space>(bvh), lsv,
                                     distCap] __device__(int vi) mutable {
      auto x = verts.template pack<3>("x", vi).template cast<T>();
      for (int d = 0; d != 3; ++d) {
        verts("inds_tri", d, vi) = reinterpret_bits<T>((Ti)-1);
        verts("ws", d, vi) = 0;
      }
      if (lsv.getSignedDistance(x) <
          0) { // only operate those within the levelset
        // int id = -1;
        T dist = distCap;
        IV triInds{};
        TV ws{0, 0, 0}, tmpWs{};
        /// iterate
        int node = 0;
        int numNodes = bvh.numNodes();
        while (node != -1 && node != numNodes) {
          auto level = bvh._levels[node];
          for (; level; --level, ++node)
            if (auto d = distance(bvh.getNodeBV(node), x); d > dist)
              break;
          // leaf node check
          if (level == 0) {
            auto tri_id = bvh._auxIndices[node];
            auto tri = boueles.template pack<3>("inds", tri_id)
                           .template reinterpret_bits<int>();
            auto d = dist_pt(x, bouverts.template pack<3>("x", tri[0]),
                             bouverts.template pack<3>("x", tri[1]),
                             bouverts.template pack<3>("x", tri[2]), tmpWs);
            if (d < dist) {
              // id = tri_id;
              dist = d;
              triInds = tri;
              ws = tmpWs;
            }
            node++;
          } else
            node = bvh._auxIndices[node];
        }
        if (dist != distCap) {
          verts.template tuple<3>("inds_tri", vi) =
              triInds.template reinterpret_bits<T>();
          verts.template tuple<3>("ws", vi) = ws;
          { //
            auto t0 = bouverts.template pack<3>("x", triInds[0]);
            auto t1 = bouverts.template pack<3>("x", triInds[1]);
            auto t2 = bouverts.template pack<3>("x", triInds[2]);
            verts.template tuple<3>("x", vi) =
                t0 * ws[0] + t1 * ws[1] + t2 * ws[2];
          }
        }
      }
    });
  }
  void apply() override {
    using namespace zs;
    // auto zsverts = get_input<ZenoParticles>("ZSParticles");
    // auto &verts = zsverts->getParticles<true>();
    auto zsobjs = RETRIEVE_OBJECT_PTRS(ZenoParticles, "ZSParticles");
    auto zsls = get_input<ZenoLevelSet>("ZSLevelSet");
    auto zsbou = get_input<ZenoParticles>("ZSBoundaryPrimitive");
    const auto &bouVerts = zsbou->getParticles();
    const auto &tris = zsbou->getQuadraturePoints();
    auto dist_cap = get_input2<float>("dist_cap");
    if (dist_cap == 0)
      dist_cap = limits<T>::max();

    constexpr auto space = execspace_e::cuda;
    auto cudaPol = cuda_exec().device(0);
    auto triBvs = retrieve_bounding_volumes(cudaPol, bouVerts, "x", tris);
    bvh_t stBvh;
    stBvh.build(cudaPol, triBvs);

    for (auto &&obj : zsobjs) {
      auto &verts = obj->getParticles<true>();
      verts.append_channels(cudaPol, {{"inds_tri", 3}, {"ws", 3}});
      using basic_ls_t = typename ZenoLevelSet::basic_ls_t;
      using const_sdf_vel_ls_t = typename ZenoLevelSet::const_sdf_vel_ls_t;
      using const_transition_ls_t =
          typename ZenoLevelSet::const_transition_ls_t;
      match([&](const auto &ls) {
        if constexpr (is_same_v<RM_CVREF_T(ls), basic_ls_t>) {
          match([&](const auto &lsPtr) {
            auto lsv = get_level_set_view<execspace_e::cuda>(lsPtr);
            bindBoundary(cudaPol, lsv, verts, stBvh, bouVerts, tris, dist_cap);
          })(ls._ls);
        } else if constexpr (is_same_v<RM_CVREF_T(ls), const_sdf_vel_ls_t>) {
          match([&](auto lsv) {
            bindBoundary(cudaPol, SdfVelFieldView{lsv}, verts, stBvh, bouVerts,
                         tris, dist_cap);
          })(ls.template getView<execspace_e::cuda>());
        } else if constexpr (is_same_v<RM_CVREF_T(ls), const_transition_ls_t>) {
          match([&](auto fieldPair) {
            auto &fvSrc = std::get<0>(fieldPair);
            auto &fvDst = std::get<1>(fieldPair);
            bindBoundary(cudaPol,
                         TransitionLevelSetView{SdfVelFieldView{fvSrc},
                                                SdfVelFieldView{fvDst},
                                                ls._stepDt, ls._alpha},
                         verts, stBvh, bouVerts, tris, dist_cap);
          })(ls.template getView<zs::execspace_e::cuda>());
        }
      })(zsls->getLevelSet());
    }

    set_output("ZSParticles", get_input("ZSParticles"));
  }
};

ZENDEFNODE(BindVerticesOnBoundary, {
                                       {"ZSParticles",
                                        "ZSLevelSet",
                                        "ZSBoundaryPrimitive",
                                        {"float", "dist_cap", "0"}},
                                       {"ZSParticles"},
                                       {},
                                       {"FEM"},
                                   });

struct MoveTowards : INode {
  template <typename VertsT>
  void setupVelocity(zs::CudaExecutionPolicy &pol, VertsT &bouVerts,
                     typename ZenoParticles::particles_t &next, double dt) {
    using namespace zs;
    // update mesh verts
    bouVerts.append_channels(pol, {{"BCtarget", 3}});
    pol(Collapse{bouVerts.size()},
        [prev = proxy<execspace_e::cuda>({}, bouVerts),
         next = proxy<execspace_e::cuda>({}, next),
         dt] __device__(int pi) mutable {
          auto newX = next.pack<3>("x", pi);
          prev.template tuple<3>("BCtarget", pi) = newX;
          prev.template tuple<3>("v", pi) =
              (newX - prev.template pack<3>("x", pi)) / dt;
        });
  }
  void apply() override {
    using namespace zs;
    fmt::print(fg(fmt::color::green), "begin executing MoveTowards\n");

    std::shared_ptr<ZenoParticles> zsprimseq{};

    if (!has_input<ZenoParticles>("ZSParticles"))
      throw std::runtime_error(
          fmt::format("no incoming prim for prim sequence!\n"));
    auto next = get_input<ZenoParticles>("ZSParticles");
    if (!next->asBoundary)
      throw std::runtime_error(
          fmt::format("incoming prim is not used as a boundary!\n"));

    auto cudaPol = cuda_exec().device(0);
    if (has_input<ZenoParticles>("ZSBoundaryPrimitive")) {
      zsprimseq = get_input<ZenoParticles>("ZSBoundaryPrimitive");
      auto numV = zsprimseq->numParticles();
      auto numE = zsprimseq->numElements();
      if (numV != next->numParticles() || numE != next->numElements()) {
        fmt::print("current numVerts {}, numEles ({}).\nIncoming "
                   "boundary primitive numVerts ({}), numEles ({})\n",
                   numV, numE, next->numParticles(), next->numElements());
        throw std::runtime_error(
            fmt::format("prim size mismatch with current sequence prim!\n"));
      }

      auto dt = get_input2<float>("framedt"); // framedt
      /// update velocity
      if (zsprimseq->hasImage(ZenoParticles::s_particleTag))
        setupVelocity(cudaPol, zsprimseq->getParticles<true>(),
                      next->getParticles(), dt);
      setupVelocity(cudaPol, zsprimseq->getParticles(), next->getParticles(),
                    dt);
    }

    fmt::print(fg(fmt::color::cyan), "done executing MoveTowards\n");
    set_output("ZSBoundaryPrimitive", get_input("ZSBoundaryPrimitive"));
  }
};

ZENDEFNODE(MoveTowards, {
                            {"ZSBoundaryPrimitive",
                             {"float", "framedt", "0.1"},
                             "ZSParticles"},
                            {"ZSBoundaryPrimitive"},
                            {},
                            {"FEM"},
                        });

} // namespace zeno