#pragma once

#include "Structures.hpp"
#include "zensim/profile/CppTimers.hpp"

#include "../kernel/tiled_vector_ops.hpp"

namespace zeno { namespace PCG {
    // the interface the equation should have:
    using T = float;

    template<int width,typename Pol,typename SrcTileVec,typename DstTileVec>
    void copy(Pol& pol,const SrcTileVec& src,const zs::SmallString& src_tag,DstTileVec& dst,const zs::SmallString& dst_tag,int offset = 0) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        pol(zs::range(src.size()),
            [src = proxy<space>({},src),src_tag,dst = proxy<space>({},dst),dst_tag,offset] __device__(int vi) mutable {
                dst.template tuple<width>(dst_tag,vi + offset) = src.template pack<width>(src_tag,vi);
        });
    }


    template<typename Pol,typename SrcTileVec,typename DstTileVec>
    void copy(Pol& pol,const SrcTileVec& src,const zs::SmallString& src_tag,DstTileVec& dst,const zs::SmallString& dst_tag,int offset = 0) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        pol(zs::range(src.size()),
            [src = proxy<space>({},src),src_tag,dst = proxy<space>({},dst),dst_tag,offset = offset] __device__(int vi) mutable {
                dst(dst_tag,vi + offset) = src(src_tag,vi);
        });
    }


    template<int space_dim,typename Pol,typename VTileVec>
    void add(Pol& pol,VTileVec& vtemp,const zs::SmallString& src0,T a0,const zs::SmallString& src1,T a1,const zs::SmallString& dst) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        pol(range(vtemp.size()),
            [vtemp = proxy<space>({},vtemp),src0,a0,src1,a1,dst] __device__(int vi) mutable {
                vtemp.template tuple<space_dim>(dst,vi) = a0 * vtemp.template pack<space_dim>(src0,vi) + a1 * vtemp.template pack<space_dim>(src1,vi);
        });
    }

    template<typename Pol,typename VTileVec>
    void add(Pol& pol,VTileVec& vtemp,const zs::SmallString& src0,T a0,const zs::SmallString& src1,T a1,const zs::SmallString& dst) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        pol(range(vtemp.size()),
            [vtemp = proxy<space>({},vtemp),src0,a0,src1,a1,dst] __device__(int vi) mutable {
                vtemp(dst,vi) = a0 * vtemp(src0,vi) + a1 * vtemp(src1,vi);
        });
    }


    template<int space_dim,typename Pol,typename VTileVec>
    void fill(Pol& pol,VTileVec& vtemp,const zs::SmallString& tag,const zs::vec<T,space_dim>& value) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        pol(range(vtemp.size()),
            [vtemp = proxy<space>({},vtemp),tag,value] __device__(int vi) mutable {
                vtemp.template tuple<space_dim>(tag,vi) = value;
        });
    }

    template<typename T,typename Pol,typename VTileVec>
    void fill(Pol& pol,VTileVec& vtemp,const zs::SmallString& tag,const T& value) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        pol(range(vtemp.size()),
            [vtemp = proxy<space>({},vtemp),tag,value] __device__(int vi) mutable {
                vtemp(tag,vi) = value;
        });
    }

    template<int space_dim,typename Pol,typename VTileVec>
    void square(Pol& pol,VTileVec& vtemp,const zs::SmallString& src,const zs::SmallString& dst) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        pol(range(vtemp.size()),
            [vtemp = proxy<space>({},vtemp),src,dst] __device__(int vi) mutable {
                vtemp.template tuple<space_dim * space_dim>(dst,vi) = 
                    vtemp.template pack<space_dim,space_dim>(src,vi).transpose() * vtemp.template pack<space_dim,space_dim>(src,vi);
        });
    }


    template<int space_dim,typename Pol,typename VTileVec>
    T inf_norm(Pol &cudaPol, VTileVec &vtemp,const zs::SmallString tag) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        Vector<T> res{vtemp.get_allocator(), 1};
        res.setVal(0);
        bool shouldSync = cudaPol.shouldSync();
        cudaPol.sync(true);
        cudaPol(range(vtemp.size()),
            [data = proxy<space>({}, vtemp), res = proxy<space>(res),tag] __device__(int pi) mutable {
                auto v = data.template pack<space_dim>(tag, pi);
                atomic_max(exec_cuda, res.data(), v.abs().max());
        });
        cudaPol.sync(shouldSync);
        return res.getVal();
    }    

    template<int simplex_dim,int space_dim,typename Pol,typename VBufTileVec,typename EBufTileVec>
    void prepare_block_diagonal_preconditioner(Pol &pol,const zs::SmallString& HTag,const EBufTileVec& etemp,const zs::SmallString& PTag,VBufTileVec& vtemp,bool use_block = true) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        // pol(zs::range(vtemp.size()),
        //     [vtemp = proxy<space>({}, vtemp),PTag] ZS_LAMBDA (int vi) mutable {
        //         constexpr int block_size = space_dim * space_dim;
        //         vtemp.template tuple<block_size>(PTag, vi) = zs::vec<T,space_dim,space_dim>::zeros();
        // });
        TILEVEC_OPS::fill(pol,vtemp,PTag,(T)0.0);

        pol(zs::range(etemp.size()),
                    [vtemp = proxy<space>({},vtemp),etemp = proxy<space>({},etemp),HTag,PTag,use_block]
                        ZS_LAMBDA(int ei) mutable{
            constexpr int h_width = space_dim * simplex_dim;
            auto inds = etemp.template pack<simplex_dim>("inds",ei).template reinterpret_bits<int>();
            for(int i = 0;i != simplex_dim;++i)
                if(inds[i] < 0)
                    return;

            auto H = etemp.template pack<h_width,h_width>(HTag,ei);

            for(int vi = 0;vi != simplex_dim;++vi)
                for(int j = 0;j != space_dim;++j){
                    if(use_block)
                        for(int k = 0;k != space_dim;++k)
                            atomic_add(exec_cuda,&vtemp(PTag,j*space_dim + k,inds[vi]),(T)H(vi*space_dim + j,vi*space_dim + k));
                    else
                        atomic_add(exec_cuda,&vtemp(PTag,j*space_dim + j,inds[vi]),(T)H(vi*space_dim + j,vi*space_dim + j));
                }
        });
        pol(zs::range(vtemp.size()),
            [vtemp = proxy<space>({},vtemp),PTag] ZS_LAMBDA(int vi) mutable{
                vtemp.template tuple<space_dim * space_dim>(PTag,vi) = inverse(vtemp.template pack<space_dim,space_dim>(PTag,vi).template cast<double>());
        });            
    }


    template<int simplex_dim,int space_dim,typename Pol,typename VBufTileVec,typename EBufTileVec,typename DynTileVec>
    void prepare_block_diagonal_preconditioner(Pol &pol,const zs::SmallString& HTag,const EBufTileVec& etemp,
            const DynTileVec& dtemp,
            const zs::SmallString& PTag,VBufTileVec& vtemp,bool use_block = true) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        TILEVEC_OPS::fill<space_dim * space_dim>(pol,vtemp,PTag,zs::vec<T,space_dim*space_dim>::zeros());
        // pol(zs::range(vtemp.size()),
        //     [vtemp = proxy<space>({}, vtemp),PTag] ZS_LAMBDA (int vi) mutable {
        //         constexpr int block_size = space_dim * space_dim;
        //         vtemp.template tuple<block_size>(PTag, vi) = zs::vec<T,space_dim,space_dim>::zeros();
        // });
        pol(zs::range(etemp.size()),
                    [vtemp = proxy<space>({},vtemp),etemp = proxy<space>({},etemp),HTag,PTag,use_block]
                        ZS_LAMBDA(int ei) mutable{
            constexpr int h_width = space_dim * simplex_dim;
            auto inds = etemp.template pack<simplex_dim>("inds",ei).reinterpret_bits(int_c);
            auto H = etemp.template pack<h_width,h_width>(HTag,ei);

            for(int vi = 0;vi != simplex_dim;++vi)
                for(int j = 0;j != space_dim;++j){
                    if(use_block)
                        for(int k = 0;k != space_dim;++k)
                            atomic_add(exec_cuda,&vtemp(PTag,j*space_dim + k,inds[vi]),(T)H(vi*space_dim + j,vi*space_dim + k));
                    else
                        atomic_add(exec_cuda,&vtemp(PTag,j*space_dim + j,inds[vi]),(T)H(vi*space_dim + j,vi*space_dim + j));
                }
        });


        pol(zs::range(dtemp.size()),
                    [vtemp = proxy<space>({},vtemp),dtemp = proxy<space>({},dtemp),HTag,PTag,use_block]
                        ZS_LAMBDA(int ei) mutable{
            constexpr int h_width = space_dim * simplex_dim;
            auto inds = dtemp.template pack<simplex_dim>("inds",ei).reinterpret_bits(int_c);
            for(int i = 0;i != simplex_dim;++i)
                if(inds[i] < 0)
                    return;
            auto H = dtemp.template pack<h_width,h_width>(HTag,ei);
            for(int vi = 0;vi != simplex_dim;++vi)
                for(int j = 0;j != space_dim;++j){
                    if(use_block)
                        for(int k = 0;k != space_dim;++k)
                            atomic_add(exec_cuda,&vtemp(PTag,j*space_dim + k,inds[vi]),(T)H(vi*space_dim + j,vi*space_dim + k));
                    else
                        atomic_add(exec_cuda,&vtemp(PTag,j*space_dim + j,inds[vi]),(T)H(vi*space_dim + j,vi*space_dim + j));
                }
        });

        pol(zs::range(vtemp.size()),
            [vtemp = proxy<space>({},vtemp),PTag] ZS_LAMBDA(int vi) mutable{
                vtemp.template tuple<space_dim * space_dim>(PTag,vi) = inverse(vtemp.template pack<space_dim,space_dim>(PTag,vi).template cast<double>());
        });  
    }

    static constexpr std::size_t count_warps(std::size_t n) noexcept {
      return (n + 31) / 32;
    }
    static constexpr int warp_index(int n) noexcept { return n / 32; }
    static constexpr auto warp_mask(int i, int n) noexcept {
      int k = n % 32;
      const int tail = n - k;
      if (i < tail)
        return zs::make_tuple(0xFFFFFFFFu, 32);
      return zs::make_tuple(((unsigned)(1ull << k) - 1), k);
    }
    template <typename T>
    static __forceinline__ __device__ void reduce_to(int i, int n, T val,
                                                     T &dst) {
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
    }
    template <typename Op = std::plus<T>>
    T reduce(zs::CudaExecutionPolicy &cudaPol, const zs::Vector<T> &res,
             Op op = {}) {
      using namespace zs;
      Vector<T> ret{res.get_allocator(), 1};
      bool shouldSync = cudaPol.shouldSync();
      cudaPol.sync(true);
      zs::reduce(cudaPol, std::begin(res), std::end(res), std::begin(ret), (T)0,
                 op);
      cudaPol.sync(shouldSync);
      return ret.getVal();
    }
#if 0
    template<int space_dim ,typename Pol,typename VTileVec>
    T dot(Pol &cudaPol, VTileVec &vertData,
          const zs::SmallString tag0, const zs::SmallString tag1) {
      using namespace zs;
      constexpr auto space = execspace_e::cuda;
      // Vector<double> res{vertData.get_allocator(), vertData.size()};

      Vector<T> res{vertData.get_allocator(),
                         count_warps(vertData.size())};
      zs::memset(zs::mem_device, res.data(), 0,
                 sizeof(T) * count_warps(vertData.size()));

       
      cudaPol(range(vertData.size()),
              [data = proxy<space>({}, vertData), res = proxy<space>(res), tag0,
               tag1, n = vertData.size()] __device__(int pi) mutable {
                auto v0 = data.pack<space_dim>(tag0, pi);
                auto v1 = data.pack<space_dim>(tag1, pi);
                auto v = v0.dot(v1);
                // res[pi] = v;
                reduce_to(pi, n, v, res[pi / 32]);
              });
      cudaPol.profile(true);    
      T ret = reduce(cudaPol, res, std::plus<T>{});// takes too much time here for nm_v = 10000
      cudaPol.profile(false);
      return ret;
    }
#else
    template<int space_dim ,typename Pol,typename VTileVec>
    T dot(Pol &pol, VTileVec &vtemp,const zs::SmallString tag0, const zs::SmallString tag1) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        Vector<T> res{vtemp.get_allocator(), 1};
        res.setVal(0);
        // auto tag0Offset=vtemp.getPropertyOffset(tag0);
        // auto tag1Offset=vtemp.getPropertyOffset(tag1);
        
        // pol.profile(true);
        bool shouldSync = pol.shouldSync();
        pol.sync(true);
        pol(range(vtemp.size()),
                [data = proxy<space>({}, vtemp), res = proxy<space>(res), tag0, tag1, n = vtemp.size()] __device__(int pi) mutable {
                    auto v0 = data.template pack<space_dim>(tag0,pi);
                    auto v1 = data.template pack<space_dim>(tag1,pi);
                    // atomic_add(exec_cuda, res.data(), v0.dot(v1));
                    reduce_to(pi, n, v0.dot(v1), res[0]);
                });
        pol.sync(shouldSync);
        // pol.profile(false);
        return res.getVal();
    }
#endif

    template<int space_dim,int simplex_dim,typename Pol,typename VTileVec,typename ETileVec>
    void multiply(Pol& pol,VTileVec& vtemp,const ETileVec& etemp,const zs::SmallString& H_tag,const zs::SmallString& inds_tag,const zs::SmallString& x_tag,const zs::SmallString& y_tag){
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        constexpr auto execTag = wrapv<space>{};

        // pol(range(vtemp.size()),
        //     [vtemp = proxy<space>({},vtemp),y_tag] __device__(int vi) mutable {
        //         vtemp.template tuple<space_dim>(y_tag,vi) = zs::vec<T,space_dim>::zeros();
        // });
        fill<space_dim>(pol,vtemp,y_tag,zs::vec<T,space_dim>::zeros());
        // zs::memset(zs::mem_device, res.data(), 0,
        //             sizeof(T) * count_warps(vtemp.size() * space_dim));

        // pol.profile(true);
#if 0
        pol(Collapse{etemp.size(), 32 * 4},
              [execTag, etemp = proxy<space>({}, etemp),
               vtemp = proxy<space>({}, vtemp), inds_tag,H_tag,x_tag,y_tag] ZS_LAMBDA(int eei, int tid) mutable {
                constexpr int dim = 3;
                int vid = tid / 32;
                int locid = (tid - vid * 32);
                int colid = locid % 12;
                int rowid = vid * dim + locid / 12;
                int numCols = 12 - (rowid % dim == dim - 1 ? 4 : 0);
                ;
                //etemp(H_tag, entryId, ei) * vtemp(x_tag, axisId, inds[vId])
                auto ee = etemp.template pack<simplex_dim>(inds_tag,eei).template reinterpret_bits<int>();
                for(int i = 0;i != simplex_dim;++i)
                    if(ee[i] < 0)
                        return;

                auto entryH = etemp(H_tag, rowid * 12 + colid, eei);
                auto entryDx = vtemp(x_tag, colid % dim, ee[colid / dim]);
                auto entryG = entryH * entryDx;
                if (locid >= 24 && locid <= 27) {
                  auto cid = colid + 8;
                  // colid / dim == cid / dim;
                  entryG += etemp(H_tag, rowid * 12 + cid, eei) *
                            vtemp(x_tag, cid % dim, ee[cid / dim]);
                }
                for (int iter = 1; iter <= 8; iter <<= 1) {
                  T tmp = __shfl_down_sync(0xFFFFFFFF, entryG, iter);
                  if (colid + iter < numCols)
                    entryG += tmp;
                }
                if (colid == 0)
                  atomic_add(execTag,
                             &vtemp(y_tag, rowid % dim, ee[rowid / dim]),
                             entryG);
              });
#elif 0
        auto ps = etemp.getPropertySize(H_tag);
        if (ps != 144 || simplex_dim != 4) {
            printf("????\n");
            getchar();
        }
        pol(range(etemp.size() * 144),
              [execTag, etemp = proxy<space>({}, etemp),
               vtemp = proxy<space>({}, vtemp), inds_tag,H_tag,x_tag,y_tag,
               n = etemp.size() * 144] ZS_LAMBDA(int idx) mutable {
                constexpr int dim = 3;
                __shared__ int offset;
                // directly use PCG_Solve_AX9_b2 from kemeng huang
                int ei = idx / 144;
                int entryId = idx % 144;
                int MRid = entryId / 12;
                int MCid = entryId % 12;
                int vId = MCid / dim;
                int axisId = MCid % dim;
                int GRtid = idx % 12;

                auto inds = etemp.template pack<simplex_dim>(inds_tag,ei).template reinterpret_bits<int>();
                for(int i = 0;i != simplex_dim;++i)
                    if(inds[i] < 0)
                        return;

                T rdata =
                    etemp(H_tag, entryId, ei) * vtemp(x_tag, axisId, inds[vId]);

                if (threadIdx.x == 0)
                  offset = 12 - GRtid;
                __syncthreads();

                int BRid = (threadIdx.x - offset + 12) / 12;
                int landidx = (threadIdx.x - offset) % 12;
                if (BRid == 0) {
                  landidx = threadIdx.x;
                }

                auto [mask, numValid] = warp_mask(idx, n);
                int laneId = threadIdx.x & 0x1f;
                bool bBoundary = (landidx == 0) || (laneId == 0);

                unsigned int mark =
                    __ballot_sync(mask, bBoundary); // a bit-mask
                mark = __brev(mark);
                unsigned int interval =
                    zs::math::min(__clz(mark << (laneId + 1)), 31 - laneId);

                for (int iter = 1; iter < 12; iter <<= 1) {
                  T tmp = __shfl_down_sync(mask, rdata, iter);
                  if (interval >= iter && laneId + iter < numValid)
                    rdata += tmp;
                }

                if (bBoundary)
                  atomic_add(execTag, &vtemp(y_tag, MRid % 3, inds[MRid / 3]),
                             rdata);
              });
#else
        pol(range(etemp.size()),
            [execTag,vtemp = proxy<space>({},vtemp),etemp = proxy<space>({},etemp),
                    inds_tag,H_tag,x_tag,y_tag] __device__(int ei) mutable {    
                constexpr int hessian_width = space_dim * simplex_dim;
                auto inds = etemp.template pack<simplex_dim>(inds_tag,ei).template reinterpret_bits<int>();
                for(int i = 0;i != simplex_dim;++i)
                    if(inds[i] < 0)
                        return;
                zs::vec<T,hessian_width> temp{};

                for(int i = 0;i != simplex_dim;++i)
                    for(int j = 0;j != space_dim;++j)
                        temp[i * space_dim + j] = vtemp(x_tag,j,inds[i]);

                auto He = etemp.template pack<hessian_width,hessian_width>(H_tag,ei);
                temp = He * temp;

                for(int i = 0;i != simplex_dim;++i)
                    for(int j = 0;j != space_dim;++j)
                        atomic_add(execTag,&vtemp(y_tag,j,inds[i]),temp[i*space_dim + j]);
        });
#endif
        // pol.profile(false);
    }



    // template<int space_dim,int simplex_dim,typename Pol,typename VTileVec,typename ETileVec,typename DynTileVec>
    // void multiply(Pol& pol,VTileVec& vtemp,const ETileVec& etemp,const DynTileVec& dtemp,
    //         const zs::SmallString& H_tag,const zs::SmallString& inds_tag,const zs::SmallString& x_tag,const zs::SmallString& y_tag){
    //     using namespace zs;
    //     constexpr auto space = execspace_e::cuda;
    //     constexpr auto execTag = wrapv<space>{};

    //     pol(range(etemp.size()),
    //         [execTag,vtemp = proxy<space>({},vtemp),etemp = proxy<space>({},etemp),
    //                 inds_tag,H_tag,x_tag,y_tag] __device__(int ei) mutable {    
    //             constexpr int hessian_width = space_dim * simplex_dim;
    //             auto inds = etemp.template pack<simplex_dim>(inds_tag,ei).template reinterpret_bits<int>();
    //             for(int i = 0;i != simplex_dim;++i)
    //                 if(inds[i] < 0)
    //                     return;
    //             zs::vec<T,hessian_width> temp{};

    //             for(int i = 0;i != simplex_dim;++i)
    //                 for(int j = 0;j != space_dim;++j)
    //                     temp[i * space_dim + j] = vtemp(x_tag,j,inds[i]);

    //             auto He = etemp.template pack<hessian_width,hessian_width>(H_tag,ei);
    //             temp = He * temp;

    //             for(int i = 0;i != simplex_dim;++i)
    //                 for(int j = 0;j != space_dim;++j)
    //                     atomic_add(execTag,&vtemp(y_tag,j,inds[i]),temp[i*space_dim + j]);
    //     });

    //     pol(range(dtemp.size()),
    //         [execTag,vtemp = proxy<space>({},vtemp),dtemp = proxy<space>({},dtemp),
    //                 inds_tag,H_tag,x_tag,y_tag] __device__(int ei) mutable {    
    //             constexpr int hessian_width = space_dim * simplex_dim;
    //             auto inds = dtemp.template pack<simplex_dim>(inds_tag,ei).template reinterpret_bits<int>();
    //             for(int i = 0;i != simplex_dim;++i)
    //                 if(inds[i] < 0)
    //                     return;
    //             zs::vec<T,hessian_width> temp{};

    //             for(int i = 0;i != simplex_dim;++i)
    //                 for(int j = 0;j != space_dim;++j)
    //                     temp[i * space_dim + j] = vtemp(x_tag,j,inds[i]);

    //             auto He = dtemp.template pack<hessian_width,hessian_width>(H_tag,ei);
    //             temp = He * temp;

    //             for(int i = 0;i != simplex_dim;++i)
    //                 for(int j = 0;j != space_dim;++j)
    //                     atomic_add(execTag,&vtemp(y_tag,j,inds[i]),temp[i*space_dim + j]);
    //     });

    // }

    template<int space_dim,int simplex_dim,typename Pol,typename VTileVec,typename ETileVec>
    void multiply_transpose(Pol& pol,VTileVec& vtemp,const ETileVec& etemp,const zs::SmallString& H_tag,const zs::SmallString& inds_tag,const zs::SmallString& x_tag,const zs::SmallString& y_tag){
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        constexpr auto execTag = wrapv<space>{};

        pol(range(vtemp.size()),
            [vtemp = proxy<space>({},vtemp),y_tag] __device__(int vi) mutable {
                vtemp.template tuple<space_dim>(y_tag,vi) = zs::vec<T,space_dim>::zeros();
        });

        pol(range(etemp.size()),
            [execTag,vtemp = proxy<space>({},vtemp),etemp = proxy<space>({},etemp),
                    inds_tag,H_tag,x_tag,y_tag] __device__(int ei) mutable {    
                constexpr int hessian_width = space_dim * simplex_dim;
                auto inds = etemp.template pack<simplex_dim>(inds_tag,ei).template reinterpret_bits<int>();

                for(int i = 0;i != simplex_dim;++i)
                    if(inds[i] < 0)
                        return;

                zs::vec<T,hessian_width> temp{};

                for(int i = 0;i != simplex_dim;++i)
                    for(int j = 0;j != space_dim;++j)
                        temp[i * space_dim + j] = vtemp(x_tag,j,inds[i]);

                auto He = etemp.template pack<hessian_width,hessian_width>(H_tag,ei);
                temp = He.transpose() * temp;

                for(int i = 0;i != simplex_dim;++i)
                    for(int j = 0;j != space_dim;++j)
                        atomic_add(execTag,&vtemp(y_tag,j,inds[i]),temp[i*space_dim + j]);
        });
    }


    // template<int space_dim,int simplex_dim,typename Pol,typename VTileVec,typename ETileVec,typename DynTileVec>
    // void multiply_transpose(Pol& pol,VTileVec& vtemp,const ETileVec& etemp,const DynTileVec& dtemp,
    //         const zs::SmallString& H_tag,const zs::SmallString& inds_tag,const zs::SmallString& x_tag,const zs::SmallString& y_tag){
    //     using namespace zs;
    //     constexpr auto space = execspace_e::cuda;
    //     constexpr auto execTag = wrapv<space>{};

    //     pol(range(vtemp.size()),
    //         [vtemp = proxy<space>({},vtemp),y_tag] __device__(int vi) mutable {
    //             vtemp.template tuple<space_dim>(y_tag,vi) = zs::vec<T,space_dim>::zeros();
    //     });

    //     pol(range(etemp.size()),
    //         [execTag,vtemp = proxy<space>({},vtemp),etemp = proxy<space>({},etemp),
    //                 inds_tag,H_tag,x_tag,y_tag] __device__(int ei) mutable {    
    //             constexpr int hessian_width = space_dim * simplex_dim;
    //             auto inds = etemp.template pack<simplex_dim>(inds_tag,ei).template reinterpret_bits<int>();
    //             zs::vec<T,hessian_width> temp{};

    //             for(int i = 0;i != simplex_dim;++i)
    //                 for(int j = 0;j != space_dim;++j)
    //                     temp[i * space_dim + j] = vtemp(x_tag,j,inds[i]);

    //             auto He = etemp.template pack<hessian_width,hessian_width>(H_tag,ei);
    //             temp = He.transpose() * temp;

    //             for(int i = 0;i != simplex_dim;++i)
    //                 for(int j = 0;j != space_dim;++j)
    //                     atomic_add(execTag,&vtemp(y_tag,j,inds[i]),temp[i*space_dim + j]);
    //     });

    //     pol(range(dtemp.size()),
    //         [execTag,vtemp = proxy<space>({},vtemp),dtemp = proxy<space>({},dtemp),
    //                 inds_tag,H_tag,x_tag,y_tag] __device__(int ei) mutable {    
    //             constexpr int hessian_width = space_dim * simplex_dim;
    //             auto inds = dtemp.template pack<simplex_dim>(inds_tag,ei).template reinterpret_bits<int>();

    //             for(int i = 0;i != simplex_dim;++i)
    //                 if(inds[i] < 0)
    //                     return;

    //             zs::vec<T,hessian_width> temp{};

    //             for(int i = 0;i != simplex_dim;++i)
    //                 for(int j = 0;j != space_dim;++j)
    //                     temp[i * space_dim + j] = vtemp(x_tag,j,inds[i]);

    //             auto He = dtemp.template pack<hessian_width,hessian_width>(H_tag,ei);
    //             temp = He.transpose() * temp;

    //             for(int i = 0;i != simplex_dim;++i)
    //                 for(int j = 0;j != space_dim;++j)
    //                     atomic_add(execTag,&vtemp(y_tag,j,inds[i]),temp[i*space_dim + j]);
    //     });

    // }



    template<int space_dim,typename Pol,typename VTileVec>
    void precondition(Pol& pol,VTileVec& vtemp,const zs::SmallString& P_tag,const zs::SmallString& src_tag,
            const zs::SmallString& dst_tag){
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        pol(zs::range(vtemp.size()),
            [vtemp = proxy<space>({},vtemp),src_tag,dst_tag,P_tag] __device__(int vi) {
                vtemp.template tuple<space_dim>(dst_tag,vi) = vtemp.template pack<space_dim,space_dim>(P_tag,vi) * vtemp.template pack<space_dim>(src_tag,vi);
        });
    }

    template<int space_dim,typename Pol,typename VTileVec>
    void project(Pol& pol,VTileVec& vtemp,const zs::SmallString& xtag,const zs::SmallString& btag) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        pol(range(vtemp.size()),
            [vtemp = proxy<space>({},vtemp),xtag,btag] __device__(int vi) mutable {
                if(vtemp(btag,vi) > 0)
                    vtemp.template tuple<space_dim>(xtag,vi) = zs::vec<T,space_dim>::zeros();
        });
    }

    // initialize the boundary values and tags, hessian, rhs and block diagonal preconditioner before apply this function
    template<int space_dim,int simplex_dim,typename Pol,typename VTileVec,typename ETileVec>
    int pcg_with_fixed_sol_solve(
        Pol& pol,
        VTileVec& vert_buffer,
        ETileVec& elm_buffer,
        const zs::SmallString& xtag,
        const zs::SmallString& bou_tag,
        const zs::SmallString& btag,
        const zs::SmallString& Ptag,
        const zs::SmallString& inds_tag,
        const zs::SmallString& Htag,
        T rel_accuracy,
        int max_iters,
        int recal_iter = 50
    ) /*-> std::enable_if_t<std::is_same_v<decltype(Equa.project(pol,vtemp,etemp,xtag,btag)), int>, void> 
        decltype((void)Equa.project(pol,vtemp,etemp,const zs::SmallString& tag,btag), 
                 (void)Equa.rhs(pol,vtemp,xtag,rtag),
                 (void)Equa.precondition(pol,const zs::SmallString& P_tag,const zs::SmallString& src_tag,const zs::SmallString& dst_tag)
                 ) */{
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();

        VTileVec vtemp{vert_buffer.get_allocator(),{
            {"x",space_dim},
            {"btag",1},
            {"b",space_dim},
            {"r",space_dim},
            {"P",space_dim * space_dim},
            {"temp",space_dim},
            {"p",space_dim},
            {"q",space_dim},
        },vert_buffer.size()};
        vtemp.resize(vert_buffer.size());

        // fmt::print("check point 0\n");

        copy<space_dim>(pol,vert_buffer,xtag,vtemp,"x");
        copy<1>(pol,vert_buffer,bou_tag,vtemp,"btag");
        copy<space_dim>(pol,vert_buffer,btag,vtemp,"b");
        copy<space_dim*space_dim>(pol,vert_buffer,Ptag,vtemp,"P");
        // fmt::print("check point 1\n");
        ETileVec etemp{elm_buffer.get_allocator(),{
            {"inds",simplex_dim},
            {"H",simplex_dim*space_dim*simplex_dim*space_dim}
        },elm_buffer.size()};
        etemp.resize(elm_buffer.size());
        copy<simplex_dim>(pol,elm_buffer,inds_tag,etemp,"inds");
        copy<space_dim * simplex_dim * space_dim * simplex_dim>(pol,elm_buffer,Htag,etemp,"H");

        multiply<space_dim,simplex_dim>(pol,vtemp,etemp,"H","inds","x","temp");
        // compute initial residual : b - Hx -> r
        add<space_dim>(pol,vtemp,"b",(T)1.0,"temp",(T)(-1.0),"r");
        project<space_dim>(pol,vtemp,"r","btag");
        // P * r -> q
        precondition<space_dim>(pol,vtemp,"P","r","q");
        // q -> p
        copy<space_dim>(pol,vtemp,"q",vtemp,"p");

        T zTrk = dot<space_dim>(pol,vtemp,"r","q");
        T residualPreconditionedNorm = std::sqrt(std::abs(zTrk));
        T localTol = rel_accuracy * residualPreconditionedNorm;
        // fmt::print("initial residual : {}\t{}\n",residualPreconditionedNorm,zTrk);

        int iter = 0;
        for(;iter != max_iters;++iter){
            if(zTrk < 0) {
                std::cout << "negative zTrk detected = " << zTrk << std::endl;
                fmt::print(fg(fmt::color::dark_cyan),"negative zTrk detected = {}\n",zTrk);
                throw std::runtime_error("negative zTrk detected");
            }
            if(std::isnan(zTrk)) {
                std::cout << "nan zTrk detected = " << zTrk << std::endl;
                fmt::print(fg(fmt::color::dark_cyan),"nan zTrk detected = {}\n",zTrk);
                throw std::runtime_error("nan zTrk detected");
            }
            if(residualPreconditionedNorm < localTol)
                break;
            // H * p -> tmp
            // pol.profile(true);
            // CppTimer timer;
            // timer.tick();
            multiply<space_dim,simplex_dim>(pol,vtemp,etemp,"H","inds","p","temp");
            // timer.tock("multiply time");
            // timer.tick();
            project<space_dim>(pol,vtemp,"temp","btag");
            // timer.tock("project0 time");
            // alpha = zTrk / (pHp)
            // timer.tick();
            T alpha = zTrk / dot<space_dim>(pol,vtemp,"temp","p");
            // timer.tock("dot time");
            // x += alpha * p
            // timer.tick();
            add<space_dim>(pol,vtemp,"x",(T)1.0,"p",alpha,"x");
            // r -= alpha * Hp
            add<space_dim>(pol,vtemp,"r",(T)1.0,"temp",-alpha,"r");
            // timer.tock("add time");
            // recalculate the residual to fix floating point error accumulation
            if(iter % (recal_iter + 1) == recal_iter){
                // r = b - Hx
                multiply<space_dim,simplex_dim>(pol,vtemp,etemp,"H","inds","x","temp");
                add<space_dim>(pol,vtemp,"b",(T)1.0,"temp",(T)(-1.0),"r");
                project<space_dim>(pol,vtemp,"r","btag");
            }
            // P * r -> q
            // timer.tick();
            precondition<space_dim>(pol,vtemp,"P","r","q");
            // timer.tock("precondition time");
            project<space_dim>(pol,vtemp,"q","btag");
            auto zTrkLast = zTrk;
            zTrk = dot<space_dim>(pol,vtemp,"q","r");
            auto beta = zTrk / zTrkLast;
            // q + beta * p -> p
            add<space_dim>(pol,vtemp,"q",(T)(1.0),"p",beta,"p");
            residualPreconditionedNorm = std::sqrt(std::abs(zTrk));
            // pol.profile(false);

            ++iter;
        }
        copy<space_dim>(pol,vtemp,"x",vert_buffer,xtag);

        return iter;
    }


    // initialize the boundary values and tags, hessian, rhs and block diagonal preconditioner before apply this function
    template<int space_dim,int simplex_dim,typename Pol,typename VTileVec,typename ETileVec,typename DTileVec>
    int pcg_with_fixed_sol_solve(
        Pol& pol,
        VTileVec& vert_buffer,
        ETileVec& elm_buffer,
        const DTileVec& dyn_buffer,
        const zs::SmallString& xtag,
        const zs::SmallString& bou_tag,
        const zs::SmallString& btag,
        const zs::SmallString& Ptag,
        const zs::SmallString& inds_tag,
        const zs::SmallString& Htag,
        T rel_accuracy,
        int max_iters,
        int recal_iter = 50
    ) /*-> std::enable_if_t<std::is_same_v<decltype(Equa.project(pol,vtemp,etemp,xtag,btag)), int>, void> 
        decltype((void)Equa.project(pol,vtemp,etemp,const zs::SmallString& tag,btag), 
                 (void)Equa.rhs(pol,vtemp,xtag,rtag),
                 (void)Equa.precondition(pol,const zs::SmallString& P_tag,const zs::SmallString& src_tag,const zs::SmallString& dst_tag)
                 ) */{
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();

        VTileVec vtemp{vert_buffer.get_allocator(),{
            {"x",space_dim},
            {"btag",1},
            {"b",space_dim},
            {"r",space_dim},
            {"P",space_dim * space_dim},
            {"temp",space_dim},
            {"p",space_dim},
            {"q",space_dim},
        },vert_buffer.size()};
        vtemp.resize(vert_buffer.size());

        // fmt::print("check point 0\n");

        copy<space_dim>(pol,vert_buffer,xtag,vtemp,"x");
        copy<1>(pol,vert_buffer,bou_tag,vtemp,"btag");
        copy<space_dim>(pol,vert_buffer,btag,vtemp,"b");
        copy<space_dim*space_dim>(pol,vert_buffer,Ptag,vtemp,"P");
        // fmt::print("check point 1\n");
        ETileVec etemp{elm_buffer.get_allocator(),{
            {"inds",simplex_dim},
            {"H",simplex_dim*space_dim*simplex_dim*space_dim}
        },elm_buffer.size() + dyn_buffer.size()};
        // etemp.resize(elm_buffer.size());
        copy<simplex_dim>(pol,elm_buffer,inds_tag,etemp,"inds");
        copy<simplex_dim>(pol,dyn_buffer,inds_tag,etemp,"inds",elm_buffer.size());
        copy<space_dim * simplex_dim * space_dim * simplex_dim>(pol,elm_buffer,Htag,etemp,"H");
        copy<space_dim * simplex_dim * space_dim * simplex_dim>(pol,dyn_buffer,Htag,etemp,"H",elm_buffer.size());

        multiply<space_dim,simplex_dim>(pol,vtemp,etemp,"H","inds","x","temp");
        // compute initial residual : b - Hx -> r
        add<space_dim>(pol,vtemp,"b",(T)1.0,"temp",(T)(-1.0),"r");
        project<space_dim>(pol,vtemp,"r","btag");
        // P * r -> q
        precondition<space_dim>(pol,vtemp,"P","r","q");
        // q -> p
        copy<space_dim>(pol,vtemp,"q",vtemp,"p");

        T zTrk = dot<space_dim>(pol,vtemp,"r","q");
        T residualPreconditionedNorm = std::sqrt(std::abs(zTrk));
        T localTol = rel_accuracy * residualPreconditionedNorm;
        // fmt::print("initial residual : {}\t{}\n",residualPreconditionedNorm,zTrk);

        int iter = 0;
        for(;iter != max_iters;++iter){
            if(zTrk < 0) {
                fmt::print(fg(fmt::color::dark_cyan),"negative zTrk detected = {}\n",zTrk);
                throw std::runtime_error("negative zTrk detected");
            }
            if(residualPreconditionedNorm < localTol)
                break;
            // H * p -> tmp
            // pol.profile(true);
            // CppTimer timer;
            // timer.tick();
            multiply<space_dim,simplex_dim>(pol,vtemp,etemp,"H","inds","p","temp");
            // timer.tock("multiply time");
            // timer.tick();
            project<space_dim>(pol,vtemp,"temp","btag");
            // timer.tock("project0 time");
            // alpha = zTrk / (pHp)
            // timer.tick();
            T alpha = zTrk / dot<space_dim>(pol,vtemp,"temp","p");
            // timer.tock("dot time");
            // x += alpha * p
            // timer.tick();
            add<space_dim>(pol,vtemp,"x",(T)1.0,"p",alpha,"x");
            // r -= alpha * Hp
            add<space_dim>(pol,vtemp,"r",(T)1.0,"temp",-alpha,"r");
            // timer.tock("add time");
            // recalculate the residual to fix floating point error accumulation
            if(iter % (recal_iter + 1) == recal_iter){
                // r = b - Hx
                multiply<space_dim,simplex_dim>(pol,vtemp,etemp,"H","inds","x","temp");
                add<space_dim>(pol,vtemp,"b",(T)1.0,"temp",(T)(-1.0),"r");
                project<space_dim>(pol,vtemp,"r","btag");
            }
            // P * r -> q
            // timer.tick();
            precondition<space_dim>(pol,vtemp,"P","r","q");
            // timer.tock("precondition time");
            project<space_dim>(pol,vtemp,"q","btag");
            auto zTrkLast = zTrk;
            zTrk = dot<space_dim>(pol,vtemp,"q","r");
            auto beta = zTrk / zTrkLast;
            // q + beta * p -> p
            add<space_dim>(pol,vtemp,"q",(T)(1.0),"p",beta,"p");
            residualPreconditionedNorm = std::sqrt(std::abs(zTrk));
            // pol.profile(false);

            ++iter;
        }
        copy<space_dim>(pol,vtemp,"x",vert_buffer,xtag);

        return iter;
    }


    // initialize the boundary values and tags, hessian, rhs and block diagonal preconditioner before apply this function
    template<int space_dim,int simplex_dim,typename Pol,typename VTileVec,typename ETileVec>
    int gn_pcg_with_fixed_sol_solve(
        Pol& pol,
        VTileVec& vert_buffer,
        ETileVec& elm_buffer,
        const zs::SmallString& xtag,
        const zs::SmallString& bou_tag,
        const zs::SmallString& btag,
        const zs::SmallString& Ptag,
        const zs::SmallString& inds_tag,
        const zs::SmallString& Htag,
        T rel_accuracy,
        int max_iters,
        int recal_iter = 50
    ) /*-> std::enable_if_t<std::is_same_v<decltype(Equa.project(pol,vtemp,etemp,xtag,btag)), int>, void> 
        decltype((void)Equa.project(pol,vtemp,etemp,const zs::SmallString& tag,btag), 
                 (void)Equa.rhs(pol,vtemp,xtag,rtag),
                 (void)Equa.precondition(pol,const zs::SmallString& P_tag,const zs::SmallString& src_tag,const zs::SmallString& dst_tag)
                 ) */{
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();

        VTileVec vtemp{vert_buffer.get_allocator(),{
            {"x",space_dim},
            {"btag",1},
            {"b",space_dim},
            {"r",space_dim},
            {"P",space_dim * space_dim},
            {"tmp0",space_dim},
            {"tmp1",space_dim},
            {"p",space_dim},
            {"q",space_dim},
        },vert_buffer.size()};
        vtemp.resize(vert_buffer.size());

        copy<space_dim>(pol,vert_buffer,xtag,vtemp,"x");
        copy<1>(pol,vert_buffer,bou_tag,vtemp,"btag");
        copy<space_dim>(pol,vert_buffer,btag,vtemp,"b");
        copy<space_dim*space_dim>(pol,vert_buffer,Ptag,vtemp,"P");
        square<space_dim>(pol,vtemp,"P","P");

        ETileVec etemp{elm_buffer.get_allocator(),{
            {"inds",simplex_dim},
            {"H",simplex_dim*space_dim*simplex_dim*space_dim}
        },elm_buffer.size()};
        etemp.resize(elm_buffer.size());
        copy<simplex_dim>(pol,elm_buffer,inds_tag,etemp,"inds");
        copy<space_dim * simplex_dim * space_dim * simplex_dim>(pol,elm_buffer,Htag,etemp,"H");

        multiply<space_dim,simplex_dim>(pol,vtemp,etemp,"H","inds","x","tmp0");
        multiply_transpose<space_dim,simplex_dim>(pol,vtemp,etemp,"H","inds","tmp0","tmp1");
        // compute initial residual : b - HTHx -> r
        add<space_dim>(pol,vtemp,"b",(T)1.0,"tmp1",(T)(-1.0),"r");
        project<space_dim>(pol,vtemp,"r","btag");
        // P * r -> q
        precondition<space_dim>(pol,vtemp,"P","r","q");
        // q -> p
        copy<space_dim>(pol,vtemp,"q",vtemp,"p");

        T zTrk = dot<space_dim>(pol,vtemp,"r","q");
        T residualPreconditionedNorm = std::sqrt(zTrk);
        T localTol = rel_accuracy * residualPreconditionedNorm;
        fmt::print("initial residual : {}\t{}\n",residualPreconditionedNorm,zTrk);

        int iter = 0;
        for(;iter != max_iters;++iter){
            if(zTrk < 0)
                throw std::runtime_error("negative zTrk detected");
            if(residualPreconditionedNorm < localTol)
                break;
            // HTH * p -> tmp
            multiply<space_dim,simplex_dim>(pol,vtemp,etemp,"H","inds","p","tmp0");
            multiply_transpose<space_dim,simplex_dim>(pol,vtemp,etemp,"H","inds","tmp0","tmp1");            
            project<space_dim>(pol,vtemp,"tmp1","btag");
            // alpha = zTrk / (pHp)
            T alpha = zTrk / dot<space_dim>(pol,vtemp,"tmp1","p");
            // x += alpha * p
            add<space_dim>(pol,vtemp,"x",(T)1.0,"p",alpha,"x");
            // r -= alpha * Hp
            add<space_dim>(pol,vtemp,"r",(T)1.0,"tmp1",-alpha,"r");
            // recalculate the residual to fix floating point error accumulation
            if(iter % (recal_iter + 1) == recal_iter){
                // r = b - Hx
                multiply<space_dim,simplex_dim>(pol,vtemp,etemp,"H","inds","x","tmp0");
                multiply_transpose<space_dim,simplex_dim>(pol,vtemp,etemp,"H","inds","tmp0","tmp1");
                add<space_dim>(pol,vtemp,"b",(T)1.0,"tmp1",(T)(-1.0),"r");
                project<space_dim>(pol,vtemp,"r","btag");
            }
            // P * r -> q
            precondition<space_dim>(pol,vtemp,"P","r","q");
            project<space_dim>(pol,vtemp,"q","btag");
            auto zTrkLast = zTrk;
            zTrk = dot<space_dim>(pol,vtemp,"q","r");
            auto beta = zTrk / zTrkLast;
            // q + beta * p -> p
            add<space_dim>(pol,vtemp,"q",(T)(1.0),"p",beta,"p");
            residualPreconditionedNorm = std::sqrt(zTrk);
            ++iter;
        }
        copy<space_dim>(pol,vtemp,"x",vert_buffer,xtag);
        fmt::print("final residual : {}\t{}\n",residualPreconditionedNorm,zTrk);

        return iter;
    }

};
};