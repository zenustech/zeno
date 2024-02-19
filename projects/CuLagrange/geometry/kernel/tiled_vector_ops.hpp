#pragma once

#include "Structures.hpp"
#include "zensim/profile/CppTimers.hpp"

namespace zeno { namespace TILEVEC_OPS {
    // the interface the equation should have:
    using T = float;

    // template<typename Pol,typename SrcTileVec

    template<int width,typename Pol,typename SrcTileVec,typename DstTileVec>
    void copy(Pol& pol,const SrcTileVec& src,const zs::SmallString& src_tag,DstTileVec& dst,const zs::SmallString& dst_tag,int offset = 0) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        // if(src.size() != dst.size())
        //     throw std::runtime_error("copy_ops_error::the size of src and dst not match");

        pol(zs::range(src.size()),
            [src = proxy<space>({},src),src_tag,dst = proxy<space>({},dst),dst_tag,offset] __device__(int vi) mutable {
                dst.template tuple<width>(dst_tag,vi + offset) = src.template pack<width>(src_tag,vi);
        });
    }

    template<typename Pol,typename SrcTileVec,typename DstTileVec>
    void copy(Pol& pol,const SrcTileVec& src,const zs::SmallString& src_tag,DstTileVec& dst,const zs::SmallString& dst_tag,int offset = 0) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        // if(src.size() != dst.size())
        //     throw std::runtime_error("copy_ops_error::the size of src and dst not match");
        if(!src.hasProperty(src_tag)){
            fmt::print(fg(fmt::color::red),"copy_ops_error::the src has no specified channel {}\n",src_tag);
            throw std::runtime_error("copy_ops_error::the src has no specified channel");
        }
        if(!dst.hasProperty(dst_tag)){
            fmt::print(fg(fmt::color::red),"copy_ops_error::the dst has no specified channel {}\n",dst_tag);
            throw std::runtime_error("copy_ops_error::the dst has no specified channel");
        }
        auto space_dim = src.getPropertySize(src_tag);
        if(dst.getPropertySize(dst_tag) != space_dim){
            std::cout << "invalid channel[" << src_tag << "] and [" << dst_tag << "] size : " << space_dim << "\t" << dst.getPropertySize(dst_tag) << std::endl;
            throw std::runtime_error("copy_ops_error::the channel size of src and dst not match");
        }
        pol(zs::range(src.size()),
            [src = proxy<space>({},src),src_tag,dst = proxy<space>({},dst),dst_tag,offset,space_dim] __device__(int vi) mutable {
                for(int i = 0;i != space_dim;++i)
                    dst(dst_tag,i,vi + offset) = src(src_tag,i,vi);
        });
    }


    template<typename Pol,typename SrcTileVec,typename DstTileVec>
    void copy(Pol& pol,const SrcTileVec& src,const int& src_tag_offset,DstTileVec& dst,const int& dst_tag_offset,int offset = 0) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        // if(src.size() != dst.size())
        //     throw std::runtime_error("copy_ops_error::the size of src and dst not match");
        // if(!src.hasProperty(src_tag)){
        //     fmt::print(fg(fmt::color::red),"copy_ops_error::the src has no specified channel {}\n",src_tag);
        //     throw std::runtime_error("copy_ops_error::the src has no specified channel");
        // }
        // if(!dst.hasProperty(dst_tag)){
        //     fmt::print(fg(fmt::color::red),"copy_ops_error::the dst has no specified channel {}\n",dst_tag);
        //     throw std::runtime_error("copy_ops_error::the dst has no specified channel");
        // }
        auto space_dim = src.getPropertySize(src_tag_offset);
        // if(dst.getPropertySize(dst_tag) != space_dim){
        //     std::cout << "invalid channel[" << src_tag << "] and [" << dst_tag << "] size : " << space_dim << "\t" << dst.getPropertySize(dst_tag) << std::endl;
        //     throw std::runtime_error("copy_ops_error::the channel size of src and dst not match");
        // }
        pol(zs::range(src.size()),
            [src = proxy<space>({},src),src_tag_offset,dst = proxy<space>({},dst),dst_tag_offset,offset,space_dim] __device__(int vi) mutable {
                for(int d = 0;d != space_dim;++d)
                    dst(dst_tag_offset + d,vi + offset) = src(src_tag_offset + d,vi);
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
                vtemp.tuple(dim_c<space_dim>,tag,vi) = value;
        });
    }


    template<typename T,typename Pol,typename VTileVec>
    void fill(Pol& pol,VTileVec& vtemp,const zs::SmallString& tag,const T& value) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        int space_dim = vtemp.getPropertySize(tag);
        pol(range(vtemp.size()),
            [vtemp = proxy<space>({},vtemp),tag,value,space_dim] __device__(int vi) mutable {
                for(int i= 0;i != space_dim;++i)
                    vtemp(tag,i,vi) = value;
        });
    }

    template<typename T,typename Pol,typename VTileVec>
    void fill(Pol& pol,VTileVec& vtemp,const int& tagOffset,const T& value) {
        using namespace zs;
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
        auto space_dim = vtemp.getPropertySize(tagOffset);

        pol(range(vtemp.size()),
            [vtemp = proxy<space>({},vtemp),tagOffset,value,space_dim] ZS_LAMBDA(int vi) mutable {
                for(int d= 0;d != space_dim;++d)
                    vtemp(tagOffset + d,vi) = value;
        });
    }


    template<int space_dim,typename Pol,typename VTileVec>
    void fill_range(Pol& pol,VTileVec& vtemp,const zs::SmallString& tag,const zs::vec<T,space_dim>& value,int start,int length) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        pol(range(length),
            [vtemp = proxy<space>({},vtemp),tag,value,start] __device__(int vi) mutable {
                vtemp.template tuple<space_dim>(tag,vi + start) = value;
        });
    }


    template<typename T,typename Pol,typename VTileVec>
    void fill_range(Pol& pol,VTileVec& vtemp,const zs::SmallString& tag,const T& value,int start,int length) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        int space_dim = vtemp.getPropertySize(tag);
        pol(range(length),
            [vtemp = proxy<space>({},vtemp),tag,value,space_dim,start] __device__(int vi) mutable {
                for(int i= 0;i != space_dim;++i)
                    vtemp(tag,i,vi + start) = value;
        });
    }



    template<typename Pol,typename SrcTileVec,typename DstTileVec>
    void assemble(Pol& pol,
        const SrcTileVec& src,const zs::SmallString& srcTag,const zs::SmallString& srcTopoTag,
        DstTileVec& dst,const zs::SmallString& dstTag) {
            using namespace zs;
            constexpr auto space = execspace_e::cuda;

            if(!src.hasProperty(srcTopoTag))
                throw std::runtime_error("tiledvec_ops::assemble::invalid src's topo channel");
            if(!src.hasProperty(srcTag))
                throw std::runtime_error("tiledvec_ops::assemble::src has no 'srcTag' channel");
            if(!dst.hasProperty(dstTag))
                throw std::runtime_error("tiledvec_ops::assemble::dst has no 'dstTag' channel");

            int simplex_size = src.getPropertySize(srcTopoTag);
            int src_space_dim = src.getPropertySize(srcTag);
            int dst_space_dim = dst.getPropertySize(dstTag);

            if(dst_space_dim * simplex_size != src_space_dim)
                throw std::runtime_error("tiledvec_ops::assemble::src_space_dim and dst_space_dim not match");

            // std::cout << "simplex_size : " << simplex_size << std::endl;
            // std::cout << "space_dim : " << space_dim << std::endl;
            // std::cout << "src_size : " << src.size() << std::endl;
            // std::cout << "dst_size : " << dst.size() << std::endl;


            pol(range(src.size()),
                [src = proxy<space>({},src),dst = proxy<space>({},dst),srcTag,srcTopoTag,dstTag,simplex_size,src_space_dim,dst_space_dim] __device__(int si) mutable {
                    for(int i = 0;i != simplex_size;++i){
                        auto idx = reinterpret_bits<int>(src(srcTopoTag,i,si));
                        if(idx < 0)
                            return;
                    }

                    for(int i = 0;i != simplex_size;++i){
                        auto idx = reinterpret_bits<int>(src(srcTopoTag,i,si));
                        for(int d = 0;d != dst_space_dim;++d){
                            atomic_add(exec_cuda,&dst(dstTag,d,idx),src(srcTag,i * dst_space_dim + d,si));
                        }
                    }
            });
    }

    template<typename Pol,typename SrcTileVec,typename DstTileVec>
    void assemble_range(Pol& pol,
        const SrcTileVec& src,const zs::SmallString& srcTag,const zs::SmallString& srcTopoTag,
        DstTileVec& dst,const zs::SmallString& dstTag,int start,int alen) {
            using namespace zs;
            constexpr auto space = execspace_e::cuda;

            if(!src.hasProperty(srcTopoTag))
                throw std::runtime_error("tiledvec_ops::assemble::invalid src's topo channel");
            if(!src.hasProperty(srcTag))
                throw std::runtime_error("tiledvec_ops::assemble::src has no 'srcTag' channel");
            if(!dst.hasProperty(dstTag))
                throw std::runtime_error("tiledvec_ops::assemble::dst has no 'dstTag' channel");

            int simplex_size = src.getPropertySize(srcTopoTag);
            int src_space_dim = src.getPropertySize(srcTag);
            int dst_space_dim = dst.getPropertySize(dstTag);


            if(dst_space_dim * simplex_size != src_space_dim)
                throw std::runtime_error("tiledvec_ops::assemble::src_space_dim and dst_space_dim not match");

            pol(range(alen),
                [src = proxy<space>({},src),dst = proxy<space>({},dst),srcTag,srcTopoTag,dstTag,start,simplex_size,space_dim = dst_space_dim] __device__(int si) mutable {
                    for(int i = 0;i != simplex_size;++i){
                        auto idx = reinterpret_bits<int>(src(srcTopoTag,i,si + start));
                        if(idx < 0)
                            return;
                    }
                    for(int i = 0;i != simplex_size;++i){
                            auto idx = reinterpret_bits<int>(src(srcTopoTag,i,si + start));
                            for(int d = 0;d != space_dim;++d){
                                atomic_add(exec_cuda,&dst(dstTag,d,idx),src(srcTag,i * space_dim + d,si + start));
                            }
                    }
            });
    }



    template<typename Pol,typename SrcTileVec,typename DstTileVec>
    void assemble(Pol& pol,
        const SrcTileVec& src,const zs::SmallString& srcTag,
        DstTileVec& dst,const zs::SmallString& dstTag) {
            using namespace zs;
            constexpr auto space = execspace_e::cuda;

            // TILEVEC_OPS::fill<space_dim>(pol,dst,"dir",zs::vec<T,space_dim>::uniform((T)0.0));

            // if(!src.hasProperty("inds") || src.getPropertySize("inds") != simplex_size)
            //     throw std::runtime_error("tiledvec_ops::assemble::invalid src's topo channel inds");

            // pol(range(src.size()),
            //     [src = proxy<space>({},src),dst = proxy<space>({},dst),src_tag,dst_tag] __device__(int si) mutable {
            //         auto inds = src.template pack<simplex_size>("inds",si).reinterpret_bits(int_c);
            //         for(int i = 0;i != simplex_size;++i)
            //             if(inds[i] < 0)
            //                 return;
            //         auto data = src.template pack<space_dim * simplex_size>(src_tag,si);
            //         for(int i = 0;i != simplex_size;++i)
            //                 for(int d = 0;d != space_dim;++d)
            //                     atomic_add(exec_cuda,&dst(dst_tag,d,inds[i]),data[i*space_dim + d]);
            // });

            assemble(pol,src,srcTag,"inds",dst,dstTag);
    }


    template<typename Pol,typename SrcTileVec,typename DstTileVec,typename DstTopoTileVec>
    void assemble_from(Pol& pol,
        const SrcTileVec& src,const zs::SmallString& srcTag,
        DstTileVec& dst,const zs::SmallString& dstTag,
        const DstTopoTileVec& topo,const zs::SmallString& dstTopoTag) {
            using namespace zs;
            constexpr auto space = execspace_e::cuda;

            if(!topo.hasProperty(dstTopoTag))
                throw std::runtime_error("tiledvec_ops::assemble_from::invalid dst's topo channel");
            if(!src.hasProperty(srcTag))
                throw std::runtime_error("tiledvec_ops::assemble::src has no 'srcTag' channel");
            if(!dst.hasProperty(dstTag))
                throw std::runtime_error("tiledvec_ops::assemble::dst has no 'dstTag' channel");
            if(dst.size() != topo.size())
                throw std::runtime_error("tiledvec_ops::assemble::dst and topo size not match");

            int simplex_size = topo.getPropertySize(dstTopoTag);
            int space_dim = src.getPropertySize(srcTag);

            pol(zs::range(dst.size()),
                [dst = proxy<space>({},dst),src = proxy<space>({},src),srcTag,dstTag,topo = proxy<space>({},topo),dstTopoTag,simplex_size,space_dim] __device__(int di) mutable {     
                    for(int i = 0;i != simplex_size;++i){
                        auto idx = reinterpret_bits<int>(topo(dstTopoTag,i,di));
                        for(int d = 0;d != space_dim;++d)
                            dst(dstTag,d,di) += src(srcTag,d,idx);
                    }
            });

    }

    template<typename Pol,typename SrcTileVec0,typename SrcTileVec1,typename DstTileVec>
    void concatenate_two_tiled_vecs(Pol& pol,
        const SrcTileVec0& src0,
        const SrcTileVec1& src1,
        DstTileVec& dst,
        const std::vector<zs::PropertyTag>& tags) {
            using namespace zs;
            constexpr auto space = execspace_e::cuda;

            for(int i = 0;i != tags.size();++i){
                auto name = tags[i].name;
                auto numChannels = tags[i].numChannels;

                if(!src0.hasProperty(name) || src0.getPropertySize() != numChannels)
                    throw std::runtime_error("concatenate_two_tiled_vecs::src0's channels not aligned with specified tags");
                if(!src1.hasProperty(name) || src1.getPropertySize() != numChannels)
                    throw std::runtime_error("concatenate_two_tiled_vecs::src1's channels not aligned with specified tags");
                if(!dst.hasProperty(name) || dst.getPropertySize() != numChannels)
                    throw std::runtime_error("concatenate_two_tiled_vecs::dst's channels not aligned with specified tags");
                if(dst.size() != (src0.size() + src1.size()))
                    throw std::runtime_error("concatenate_two_tiled_vecs::dst.size() != src0.size() + src1.size()");
            }

            for(int i = 0;i != tags.size();++i) {
                auto name = tags[i].name;
                auto numChannels = tags[i].numChannels;
                copy(pol,src0,name,dst,name,0);
                copy(pol,src1,name,dst,name,src0.size());
            }
    }


    template<int space_dim,int simplex_size,typename Pol,typename SrcTileVec,typename DstTileVec>
    void assemble_from(Pol& pol,
        const SrcTileVec& src,const zs::SmallString& srcTag,
        DstTileVec& dst,const zs::SmallString& dstTag,const zs::SmallString& dstTopoTag) {
            // using namespace zs;
            // constexpr auto space = execspace_e::cuda;

            // if(!dst.hasProperty(dstTopoTag) || dst.getPropertySize(dstTopoTag) != simplex_size)
            //     throw std::runtime_error("tiledvec_ops::assemble_from::invalid dst's topo channel");
            // if(!src.hasProperty(srcTag))
            //     throw std::runtime_error("tiledvec_ops::assemble::src has no 'srcTag' channel");
            // if(!dst.hasProperty(dstTag))
            //     throw std::runtime_error("tiledvec_ops::assemble::dst has no 'dstTag' channel");

            // pol(zs::range(dst.size()),
            //     [dst = proxy<space>({},dst),src = proxy<space>({},src),srcTag,dstTag,dstTopoTag] __device__(int di) mutable {
            //         auto inds = dst.template pack<simplex_size>(dstTopoTag,di).reinterpret_bits(int_c);
            //         for(int i = 0;i != simplex_size;++i)
            //             dst.template tuple<space_dim>(dstTag,di) = dst.template pack<space_dim>(dstTag,di) + src.template pack<space_dim>(srcTag,inds[i]);
            // });
            assemble_from(pol,src,srcTag,dst,dstTag,dst,dstTopoTag);

    }

    // maybe we also need a weighted assemble func

    template<int space_dim,typename Pol,typename VTileVec>
    void normalized_channel(Pol& pol,VTileVec& vtemp,const zs::SmallString& tag, T eps = 1e-6) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;

        pol(range(vtemp.size()),
            [vtemp = proxy<space>({},vtemp),tag,eps] __device__(int vi) mutable {
                auto d = vtemp.template pack<space_dim>(tag,vi);
                auto dn = d.norm();
                d = dn > eps ? d/dn : zs::vec<T,space_dim>::zeros();
                vtemp.template tuple<space_dim>(tag,vi) = d;
        });
    }

    template<int space_dim,typename Pol,typename VTileVec>
    void uniform_scale(Pol& pol,VTileVec& vtemp,const zs::SmallString& tag,T s) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;

        pol(range(vtemp.size()),
            [vtemp = proxy<space>({},vtemp),tag,s] __device__(int vi) mutable {
                vtemp.template tuple<space_dim>(tag,vi) = vtemp.template pack<space_dim>(tag,vi) * s;
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


    template<int space_dim,typename Pol,typename VTileVec>
    T max_norm(Pol &cudaPol, VTileVec &vtemp,const zs::SmallString tag) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        Vector<T> res{vtemp.get_allocator(), 1};
        res.setVal(0);
        bool shouldSync = cudaPol.shouldSync();
        cudaPol.sync(true);
        cudaPol(range(vtemp.size()),
            [data = proxy<space>({}, vtemp), res = proxy<space>(res),tag] __device__(int pi) mutable {
                auto v = data.template pack<space_dim>(tag, pi);
                atomic_max(exec_cuda, res.data(), v.norm());
        });
        cudaPol.sync(shouldSync);
        return res.getVal();
    }        

    // template<int simplex_dim,int space_dim,typename Pol,typename VBufTileVec,typename EBufTileVec>
    // void prepare_block_diagonal_preconditioner(Pol &pol,const zs::SmallString& HTag,const EBufTileVec& etemp,const zs::SmallString& PTag,VBufTileVec& vtemp,bool use_block = true) {
    //     using namespace zs;
    //     constexpr auto space = execspace_e::cuda;
    //     pol(zs::range(vtemp.size()),
    //         [vtemp = proxy<space>({}, vtemp),PTag] ZS_LAMBDA (int vi) mutable {
    //             constexpr int block_size = space_dim * space_dim;
    //             vtemp.template tuple<block_size>(PTag, vi) = zs::vec<T,space_dim,space_dim>::zeros();
    //     });
    //     pol(zs::range(etemp.size()),
    //                 [vtemp = proxy<space>({},vtemp),etemp = proxy<space>({},etemp),HTag,PTag,use_block]
    //                     ZS_LAMBDA(int ei) mutable{
    //         constexpr int h_width = space_dim * simplex_dim;
    //         auto inds = etemp.template pack<simplex_dim>("inds",ei).template reinterpret_bits<int>();
    //         auto H = etemp.template pack<h_width,h_width>(HTag,ei);

    //         for(int vi = 0;vi != simplex_dim;++vi)
    //             for(int j = 0;j != space_dim;++j){
    //                 if(use_block)
    //                     for(int k = 0;k != space_dim;++k)
    //                         atomic_add(exec_cuda,&vtemp(PTag,j*space_dim + k,inds[vi]),(T)H(vi*space_dim + j,vi*space_dim + k));
    //                 else
    //                     atomic_add(exec_cuda,&vtemp(PTag,j*space_dim + j,inds[vi]),(T)H(vi*space_dim + j,vi*space_dim + j));
    //             }
    //     });
    //     pol(zs::range(vtemp.size()),
    //         [vtemp = proxy<space>({},vtemp),PTag] ZS_LAMBDA(int vi) mutable{
    //             vtemp.template tuple<space_dim * space_dim>(PTag,vi) = inverse(vtemp.template pack<space_dim,space_dim>(PTag,vi).template cast<double>());
    //     });            
    // }

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
    T dot(Pol &pol,const VTileVec &vtemp,const zs::SmallString& tag0, const zs::SmallString& tag1) {
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

    template<int space_dim ,typename Pol,typename VTileVec>
    T dot(Pol &pol,const VTileVec &vtemp,const zs::SmallString& tag0, const zs::SmallString& tag1,const size_t& len) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        Vector<T> res{vtemp.get_allocator(), 1};
        res.setVal(0);
        // auto tag0Offset=vtemp.getPropertyOffset(tag0);
        // auto tag1Offset=vtemp.getPropertyOffset(tag1);
        
        // pol.profile(true);
        bool shouldSync = pol.shouldSync();
        pol.sync(true);
        pol(range(len),
                [data = proxy<space>({}, vtemp), res = proxy<space>(res), tag0, tag1, n = vtemp.size()] __device__(int pi) mutable {
                    auto v0 = data.template pack<space_dim>(tag0,pi);
                    auto v1 = data.template pack<space_dim>(tag1,pi);
                    atomic_add(exec_cuda, res.data(), v0.dot(v1));
                    // reduce_to(pi, n, v0.dot(v1), res[0]);
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

    // tempalte<int dim>
    // void build_spacial_hash(Pol& pol,
    //     int nm_keys,
    //     const zs::Vector<zs::vec<int,dim>>& keys,
    //     zs::bht<int,dim,int>& hash,zs::Vector<int>& hash_buffer) {
    //         constexpr auto space = Pol::exec_tag::value;
    //         nm_keys = nm_keys <= 0 ? keys.size() : nm_keys;
    //         hash.reset(pol,true);
    //         pol(zs::range(nm_keys),[
    //             keys = proxy<space>(keys),
    //             hash = proxy<space>(hash),
    //             hash_buffer = proxy<space>(hash)
    //         ])
    // }
};

};