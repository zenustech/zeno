#pragma once

#include "../../Structures.hpp"

namespace zeno { namespace PCG {
    // the interface the equation should have:
    using T = float;

    template<int space_dim,typename Pol,typename VTileVec>
    T inf_norm(Pol &cudaPol, VTileVec &vtemp,const zs::SmallString tag) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        Vector<T> res{vtemp.get_allocator(), 1};
        res.setVal(0);
        cudaPol(range(vtemp.size()),
            [data = proxy<space>({}, vtemp), res = proxy<space>(res),tag] __device__(int pi) mutable {
                auto v = data.template pack<space_dim>(tag, pi);
                atomic_max(exec_cuda, res.data(), v.abs().max());
        });
        return res.getVal();
    }    

    template<int simplex_dim,int space_dim,typename Pol,typename VBufTileVec,typename EBufTileVec>
    void prepare_block_diagonal_preconditioner(Pol &pol,const zs::SmallString& HTag,const EBufTileVec& etemp,const zs::SmallString& PTag,VBufTileVec& vtemp,bool use_block = true) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        pol(zs::range(vtemp.size()),
            [vtemp = proxy<space>({}, vtemp),PTag] ZS_LAMBDA (int vi) mutable {
                constexpr int block_size = space_dim * space_dim;
                vtemp.template tuple<block_size>(PTag, vi) = zs::vec<T,space_dim,space_dim>::zeros();
        });
        pol(zs::range(etemp.size()),
                    [vtemp = proxy<space>({},vtemp),etemp = proxy<space>({},etemp),HTag,PTag,use_block]
                        ZS_LAMBDA(int ei) mutable{
            constexpr int h_width = space_dim * simplex_dim;
            auto inds = etemp.template pack<simplex_dim>("inds",ei).template reinterpret_bits<int>();
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


    template<int space_dim ,typename Pol,typename VTileVec>
    T dot(Pol &pol, VTileVec &vtemp,const zs::SmallString tag0, const zs::SmallString tag1) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        Vector<T> res{vtemp.get_allocator(), 1};
        res.setVal(0);
        pol(range(vtemp.size()),
                [data = proxy<space>({}, vtemp), res = proxy<space>(res), tag0,tag1] __device__(int pi) mutable {
                    auto v0 = data.template pack<space_dim>(tag0,pi);
                    auto v1 = data.template pack<space_dim>(tag1,pi);
                    atomic_add(exec_cuda, res.data(), v0.dot(v1));
                });
        return res.getVal();
    }

    template<int space_dim,int simplex_dim,typename Pol,typename VTileVec,typename ETileVec>
    void multiply(Pol& pol,VTileVec& vtemp,const ETileVec& etemp,const zs::SmallString& H_tag,const zs::SmallString& inds_tag,const zs::SmallString& x_tag,const zs::SmallString& y_tag){
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
                temp = He * temp;

                for(int i = 0;i != simplex_dim;++i)
                    for(int j = 0;j != space_dim;++j)
                        atomic_add(execTag,&vtemp(y_tag,j,inds[i]),temp[i*space_dim + j]);
        });
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

    template<int width,typename Pol,typename SrcTileVec,typename DstTileVec>
    void copy(Pol& pol,const SrcTileVec& src,const zs::SmallString& src_tag,DstTileVec& dst,const zs::SmallString& dst_tag) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        pol(zs::range(src.size()),
            [src = proxy<space>({},src),src_tag,dst = proxy<space>({},dst),dst_tag] __device__(int vi) {
                dst.template tuple<width>(dst_tag,vi) = src.template pack<width>(src_tag,vi);
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

    template<int space_dim,typename Pol,typename VTileVec>
    void fill(Pol& pol,VTileVec& vtemp,const zs::SmallString& tag,const zs::vec<T,space_dim>& value) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        pol(range(vtemp.size()),
            [vtemp = proxy<space>({},vtemp),tag,value] __device__(int vi) mutable {
                vtemp.template tuple<space_dim>(tag,vi) = value;
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
        T residualPreconditionedNorm = std::sqrt(zTrk);
        T localTol = rel_accuracy * residualPreconditionedNorm;
        fmt::print("initial residual : {}\t{}\n",residualPreconditionedNorm,zTrk);

        int iter = 0;
        for(;iter != max_iters;++iter){
            if(zTrk < 0)
                throw std::runtime_error("negative zTrk detected");
            if(residualPreconditionedNorm < localTol)
                break;
            // H * p -> tmp
            multiply<space_dim,simplex_dim>(pol,vtemp,etemp,"H","inds","p","temp");
            project<space_dim>(pol,vtemp,"temp","btag");
            // alpha = zTrk / (pHp)
            T alpha = zTrk / dot<space_dim>(pol,vtemp,"temp","p");
            // x += alpha * p
            add<space_dim>(pol,vtemp,"x",(T)1.0,"p",alpha,"x");
            // r -= alpha * Hp
            add<space_dim>(pol,vtemp,"r",(T)1.0,"temp",-alpha,"r");
            // recalculate the residual to fix floating point error accumulation
            if(iter % (recal_iter + 1) == recal_iter){
                // r = b - Hx
                multiply<space_dim,simplex_dim>(pol,vtemp,etemp,"H","inds","x","temp");
                add<space_dim>(pol,vtemp,"b",(T)1.0,"temp",(T)(-1.0),"r");
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