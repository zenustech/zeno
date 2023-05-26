#pragma once
#if __CUDA_ARCH__ >= 800
#include <cooperative_groups/reduce.h>
#endif

namespace zeno {

/// credits: du wenxin
template <int n = 1, typename T_ = float>
struct HessianPiece {
    using T = T_;
    using HessT = zs::vec<T, n * 3, n * 3>;
    using IndsT = zs::vec<int, n>;
    zs::Vector<HessT> hess;
    zs::Vector<IndsT> inds;
    zs::Vector<int> cnt;
    using allocator_t = typename zs::Vector<int>::allocator_type;
    void init(const allocator_t &allocator, std::size_t size = 0) {
        hess = zs::Vector<HessT>{allocator, size};
        inds = zs::Vector<IndsT>{allocator, size};
        cnt = zs::Vector<int>{allocator, 1};
    }
    int count() const {
        return cnt.getVal();
    }
    int increaseCount(int inc) {
        int v = cnt.getVal();
        cnt.setVal(v + inc);
        hess.resize((std::size_t)(v + inc));
        inds.resize((std::size_t)(v + inc));
        return v;
    }
    void reset(bool setZero = true, std::size_t count = 0) {
        if (setZero)
            hess.reset(0);
        cnt.setVal(count);
    }
};
template <typename HessianPieceT>
struct HessianView {
    static constexpr bool is_const_structure = std::is_const_v<HessianPieceT>;
    using T = typename HessianPieceT::T;
    using HT = typename HessianPieceT::HessT;
    using IT = typename HessianPieceT::IndsT;
    zs::conditional_t<is_const_structure, const HT *, HT *> hess;
    zs::conditional_t<is_const_structure, const IT *, IT *> inds;
    zs::conditional_t<is_const_structure, const int *, int *> cnt;
};
template <zs::execspace_e space, int n, typename T>
inline HessianView<HessianPiece<n, T>> proxy(HessianPiece<n, T> &hp) {
    return HessianView<HessianPiece<n, T>>{hp.hess.data(), hp.inds.data(), hp.cnt.data()};
}
template <zs::execspace_e space, int n, typename T>
inline HessianView<const HessianPiece<n, T>> proxy(const HessianPiece<n, T> &hp) {
    return HessianView<const HessianPiece<n, T>>{hp.hess.data(), hp.inds.data(), hp.cnt.data()};
}

/// utilities
constexpr std::size_t count_warps(std::size_t n) noexcept {
    return (n + 31) / 32;
}
constexpr int warp_index(int n) noexcept {
    return n / 32;
}
constexpr auto warp_mask(int i, int n) noexcept {
    int k = n % 32;
    const int tail = n - k;
    if (i < tail)
        return zs::make_tuple(0xFFFFFFFFu, 32);
    return zs::make_tuple(((unsigned)(1ull << k) - 1), k);
}

template <typename T>
__forceinline__ __device__ void reduce_to(int i, int n, T val, T &dst) {
#if __CUDA_ARCH__ >= 800
    auto tile = zs::cg::tiled_partition<32>(zs::cg::this_thread_block());
    auto ret = zs::cg::reduce(tile, val, zs::cg::plus<T>());
    if (tile.thread_rank() == 0)
        zs::atomic_add(zs::exec_cuda, &dst, ret);
#else
    auto [mask, numValid] = warp_mask(i, n);
    __syncwarp(mask);
    auto locid = threadIdx.x & 31;
    for (int stride = 1; stride < 32; stride <<= 1) {
        auto tmp = __shfl_down_sync(mask, val, stride);
        if (locid + stride < numValid)
            val += tmp;
    }
    if (locid == 0)
        zs::atomic_add(zs::exec_cuda, &dst, val);
#endif
}

template <typename T>
inline T computeHb(const T d2, const T dHat2) {
    if (d2 >= dHat2)
        return 0;
    T t2 = d2 - dHat2;
    return ((std::log(d2 / dHat2) * -2 - t2 * 4 / d2) + (t2 / d2) * (t2 / d2));
}

template <typename TileVecT, typename VecT>
inline void retrieve_points(zs::CudaExecutionPolicy &pol, const TileVecT &vtemp, const zs::SmallString &xTag,
                            const typename ZenoParticles::particles_t &eles, int voffset, zs::Vector<VecT> &ret) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    ret.resize(eles.size());
    pol(range(eles.size()), [eles = proxy<space>({}, eles), pts = proxy<space>(ret), vtemp = proxy<space>({}, vtemp),
                             xTag, voffset] ZS_LAMBDA(int ei) mutable {
        auto ind = eles("inds", ei, int_c) + voffset;
        auto x0 = vtemp.pack(dim_c<3>, xTag, ind);
        pts[ei] = x0;
    });
}
template <typename TileVecT, int codim = 3>
inline void retrieve_bounding_volumes(zs::CudaExecutionPolicy &pol, const TileVecT &vtemp, const zs::SmallString &xTag,
                                      const typename ZenoParticles::particles_t &eles, zs::wrapv<codim>, int voffset,
                                      zs::Vector<zs::AABBBox<3, typename TileVecT::value_type>> &ret) {
    using namespace zs;
    using T = typename TileVecT::value_type;
    using bv_t = AABBBox<3, T>;
    static_assert(codim >= 1 && codim <= 4, "invalid co-dimension!\n");
    constexpr auto space = execspace_e::cuda;
    ret.resize(eles.size());
    pol(range(eles.size()), [eles = proxy<space>({}, eles), bvs = proxy<space>(ret), vtemp = proxy<space>({}, vtemp),
                             codim_v = wrapv<codim>{}, xTag, voffset] ZS_LAMBDA(int ei) mutable {
        constexpr int dim = RM_CVREF_T(codim_v)::value;
        auto inds = eles.pack(dim_c<dim>, "inds", ei, int_c) + voffset;
        auto x0 = vtemp.pack(dim_c<3>, xTag, inds[0]);
        bv_t bv{x0, x0};
        for (int d = 1; d != dim; ++d)
            merge(bv, vtemp.pack(dim_c<3>, xTag, inds[d]));
        bvs[ei] = bv;
    });
}
template <typename TileVecT, int codim = 3>
inline zs::Vector<zs::AABBBox<3, typename TileVecT::value_type>>
retrieve_bounding_volumes(zs::CudaExecutionPolicy &pol, const TileVecT &vtemp, const zs::SmallString &xTag,
                          const typename ZenoParticles::particles_t &eles, zs::wrapv<codim>, int voffset) {
    using namespace zs;
    using T = typename TileVecT::value_type;
    using bv_t = AABBBox<3, T>;
    static_assert(codim >= 1 && codim <= 4, "invalid co-dimension!\n");
    constexpr auto space = execspace_e::cuda;
    zs::Vector<bv_t> ret{eles.get_allocator(), eles.size()};
    pol(range(eles.size()), [eles = proxy<space>({}, eles), bvs = proxy<space>(ret), vtemp = proxy<space>({}, vtemp),
                             codim_v = wrapv<codim>{}, xTag, voffset] ZS_LAMBDA(int ei) mutable {
        constexpr int dim = RM_CVREF_T(codim_v)::value;
        auto inds = eles.pack(dim_c<dim>, "inds", ei, int_c) + voffset;
        auto x0 = vtemp.pack(dim_c<3>, xTag, inds[0]);
        bv_t bv{x0, x0};
        for (int d = 1; d != dim; ++d)
            merge(bv, vtemp.pack(dim_c<3>, xTag, inds[d]));
        bvs[ei] = bv;
    });
    return ret;
}
template <typename TileVecT0, typename TileVecT1, int codim = 3>
inline void retrieve_bounding_volumes(zs::CudaExecutionPolicy &pol, const TileVecT0 &verts, const zs::SmallString &xTag,
                                      const typename ZenoParticles::particles_t &eles, zs::wrapv<codim>,
                                      const TileVecT1 &vtemp, const zs::SmallString &dirTag, float stepSize,
                                      int voffset, zs::Vector<zs::AABBBox<3, typename TileVecT0::value_type>> &ret) {
    using namespace zs;
    using T = typename TileVecT0::value_type;
    using bv_t = AABBBox<3, T>;
    static_assert(codim >= 1 && codim <= 4, "invalid co-dimension!\n");
    constexpr auto space = execspace_e::cuda;
    ret.resize(eles.size());
    pol(zs::range(eles.size()), [eles = proxy<space>({}, eles), bvs = proxy<space>(ret),
                                 verts = proxy<space>({}, verts), vtemp = proxy<space>({}, vtemp),
                                 codim_v = wrapv<codim>{}, xTag, dirTag, stepSize, voffset] ZS_LAMBDA(int ei) mutable {
        constexpr int dim = RM_CVREF_T(codim_v)::value;
        auto inds = eles.pack(dim_c<dim>, "inds", ei, int_c) + voffset;
        auto x0 = verts.pack(dim_c<3>, xTag, inds[0]);
        auto dir0 = vtemp.pack(dim_c<3>, dirTag, inds[0]);
        bv_t bv{get_bounding_box(x0, x0 + stepSize * dir0)};
        for (int d = 1; d != dim; ++d) {
            auto x = verts.pack(dim_c<3>, xTag, inds[d]);
            auto dir = vtemp.pack(dim_c<3>, dirTag, inds[d]);
            merge(bv, x);
            merge(bv, x + stepSize * dir);
        }
        bvs[ei] = bv;
    });
}
template <typename TileVecT0, typename TileVecT1, int codim = 3>
inline zs::Vector<zs::AABBBox<3, typename TileVecT0::value_type>>
retrieve_bounding_volumes(zs::CudaExecutionPolicy &pol, const TileVecT0 &verts, const zs::SmallString &xTag,
                          const typename ZenoParticles::particles_t &eles, zs::wrapv<codim>, const TileVecT1 &vtemp,
                          const zs::SmallString &dirTag, float stepSize, int voffset) {
    using namespace zs;
    using T = typename TileVecT0::value_type;
    using bv_t = AABBBox<3, T>;
    static_assert(codim >= 1 && codim <= 4, "invalid co-dimension!\n");
    constexpr auto space = execspace_e::cuda;
    Vector<bv_t> ret{eles.get_allocator(), eles.size()};
    pol(zs::range(eles.size()), [eles = proxy<space>({}, eles), bvs = proxy<space>(ret),
                                 verts = proxy<space>({}, verts), vtemp = proxy<space>({}, vtemp),
                                 codim_v = wrapv<codim>{}, xTag, dirTag, stepSize, voffset] ZS_LAMBDA(int ei) mutable {
        constexpr int dim = RM_CVREF_T(codim_v)::value;
        auto inds = eles.pack(dim_c<dim>, "inds", ei, int_c) + voffset;
        auto x0 = verts.pack(dim_c<3>, xTag, inds[0]);
        auto dir0 = vtemp.pack(dim_c<3>, dirTag, inds[0]);
        bv_t bv{get_bounding_box(x0, x0 + stepSize * dir0)};
        for (int d = 1; d != dim; ++d) {
            auto x = verts.pack(dim_c<3>, xTag, inds[d]);
            auto dir = vtemp.pack(dim_c<3>, dirTag, inds[d]);
            merge(bv, x);
            merge(bv, x + stepSize * dir);
        }
        bvs[ei] = bv;
    });
    return ret;
}
template <typename T, typename Op = std::plus<T>>
inline T reduce(zs::CudaExecutionPolicy &cudaPol, const zs::Vector<T> &res, Op op = {}) {
    using namespace zs;
    Vector<T> ret{res.get_allocator(), 1};
    bool shouldSync = cudaPol.shouldSync();
    cudaPol.sync(true);
    zs::reduce(cudaPol, std::begin(res), std::end(res), std::begin(ret), (T)0, op);
    cudaPol.sync(shouldSync);
    return ret.getVal();
}
template <typename T, typename Op = std::plus<T>>
inline T reduce(zs::CudaExecutionPolicy &cudaPol, const zs::Vector<T> &res, T e, Op op) {
    using namespace zs;
    Vector<T> ret{res.get_allocator(), 1};
    bool shouldSync = cudaPol.shouldSync();
    cudaPol.sync(true);
    zs::reduce(cudaPol, std::begin(res), std::end(res), std::begin(ret), e, op);
    cudaPol.sync(shouldSync);
    return ret.getVal();
}
template <typename TT, typename T, typename AllocatorT>
inline TT dot(zs::CudaExecutionPolicy &cudaPol, zs::wrapt<TT>, zs::TileVector<T, 32, AllocatorT> &vertData,
              const zs::SmallString tag0, const zs::SmallString tag1) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    Vector<TT> res{vertData.get_allocator(), count_warps(vertData.size())};
    res.reset(0);
    cudaPol(range(vertData.size()), [data = proxy<space>({}, vertData), res = proxy<space>(res), n = vertData.size(),
                                     offset0 = vertData.getPropertyOffset(tag0),
                                     offset1 = vertData.getPropertyOffset(tag1)] __device__(int pi) mutable {
        auto v0 = data.pack(dim_c<3>, offset0, pi).template cast<TT>();
        auto v1 = data.pack(dim_c<3>, offset1, pi).template cast<TT>();
        auto v = v0.dot(v1);
        reduce_to(pi, n, v, res[pi / 32]);
    });
    return reduce(cudaPol, res, thrust::plus<TT>{});
}
template <typename T, typename AllocatorT>
inline T dot(zs::CudaExecutionPolicy &cudaPol, zs::TileVector<T, 32, AllocatorT> &vertData, const zs::SmallString tag0,
             const zs::SmallString tag1) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    // Vector<double> res{vertData.get_allocator(), vertData.size()};
    Vector<T> res{vertData.get_allocator(), count_warps(vertData.size())};
    zs::memset(zs::mem_device, res.data(), 0, sizeof(T) * count_warps(vertData.size()));
    cudaPol(range(vertData.size()), [data = proxy<space>({}, vertData), res = proxy<space>(res), tag0, tag1,
                                     n = vertData.size()] __device__(int pi) mutable {
        auto v0 = data.pack(dim_c<3>, tag0, pi);
        auto v1 = data.pack(dim_c<3>, tag1, pi);
        auto v = v0.dot(v1);
        // res[pi] = v;
        reduce_to(pi, n, v, res[pi / 32]);
    });
    return reduce(cudaPol, res, thrust::plus<T>{});
}

template <typename VecT, typename VecTM, int N = VecT::template range_t<0>::value,
          zs::enable_if_all<N % 3 == 0, N == VecT::template range_t<1>::value, VecTM::dim == 2,
                            VecTM::template range_t<0>::value == 3, VecTM::template range_t<1>::value == 3> = 0>
__forceinline__ __device__ void rotate_hessian(zs::VecInterface<VecT> &H, const VecTM BCbasis[N / 3],
                                               const int BCorder[N / 3], const int BCfixed[], bool projectDBC) {
    // hessian rotation: trans^T hess * trans
    // left trans^T: multiplied on rows
    // right trans: multiplied on cols
    constexpr int NV = N / 3;
    // rotate and project
    for (int vi = 0; vi != NV; ++vi) {
        int offsetI = vi * 3;
        for (int vj = 0; vj != NV; ++vj) {
            int offsetJ = vj * 3;
            VecTM tmp{};
            for (int i = 0; i != 3; ++i)
                for (int j = 0; j != 3; ++j)
                    tmp(i, j) = H(offsetI + i, offsetJ + j);
            // rotate
            tmp = BCbasis[vi].transpose() * tmp * BCbasis[vj];
            // project
            if (projectDBC) {
                for (int i = 0; i != 3; ++i) {
                    bool clearRow = i < BCorder[vi];
                    for (int j = 0; j != 3; ++j) {
                        bool clearCol = j < BCorder[vj];
                        if (clearRow || clearCol)
                            tmp(i, j) = (vi == vj && i == j ? 1 : 0);
                    }
                }
            } else {
                for (int i = 0; i != 3; ++i) {
                    bool clearRow = i < BCorder[vi] && BCfixed[vi] == 1;
                    for (int j = 0; j != 3; ++j) {
                        bool clearCol = j < BCorder[vj] && BCfixed[vj] == 1;
                        if (clearRow || clearCol)
                            tmp(i, j) = (vi == vj && i == j ? 1 : 0);
                    }
                }
            }
            for (int i = 0; i != 3; ++i)
                for (int j = 0; j != 3; ++j)
                    H(offsetI + i, offsetJ + j) = tmp(i, j);
        }
    }
    return;
}

template <typename VecT, int N = VecT::template range_t<0>::value,
          zs::enable_if_all<N % 3 == 0, N == VecT::template range_t<1>::value> = 0>
__forceinline__ __device__ void rotate_hessian(zs::VecInterface<VecT> &H, const int BCorder[], const int BCfixed[],
                                               bool projectDBC) {
    // hessian rotation: trans^T hess * trans
    // left trans^T: multiplied on rows
    // right trans: multiplied on cols
    constexpr int NV = N / 3;
    using T = typename VecT::value_type;
    // rotate and project
    for (int vi = 0; vi != NV; ++vi) {
        int offsetI = vi * 3;
        for (int vj = 0; vj != NV; ++vj) {
            int offsetJ = vj * 3;
            using mat3 = zs::vec<T, 3, 3>;
            mat3 tmp{};
            for (int i = 0; i != 3; ++i)
                for (int j = 0; j != 3; ++j)
                    tmp(i, j) = H(offsetI + i, offsetJ + j);
            // rotate
            // tmp = BCbasis[vi].transpose() * tmp * BCbasis[vj];
            // project
            if (projectDBC) {
                for (int i = 0; i != 3; ++i) {
                    bool clearRow = i < BCorder[vi];
                    for (int j = 0; j != 3; ++j) {
                        bool clearCol = j < BCorder[vj];
                        if (clearRow || clearCol)
                            tmp(i, j) = (vi == vj && i == j ? 1 : 0);
                    }
                }
            } else {
                for (int i = 0; i != 3; ++i) {
                    bool clearRow = i < BCorder[vi] && BCfixed[vi] == 1;
                    for (int j = 0; j != 3; ++j) {
                        bool clearCol = j < BCorder[vj] && BCfixed[vj] == 1;
                        if (clearRow || clearCol)
                            tmp(i, j) = (vi == vj && i == j ? 1 : 0);
                    }
                }
            }
            for (int i = 0; i != 3; ++i)
                for (int j = 0; j != 3; ++j)
                    H(offsetI + i, offsetJ + j) = tmp(i, j);
        }
    }
    return;
}

} // namespace zeno