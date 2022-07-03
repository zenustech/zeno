#pragma once

#include "../../Structures.hpp"

namespace zeno {
    // the interface the equation should have:
    using T = float;

    template<int simplex_dim,typename Pol,typename ElmTileVec,typename VBufTileVec,typename EBufTileVec>
    void prepare_preconditioner(Pol &pol,const ElmTileVec& eles,const zs::SmallString& HTag,const EBufTileVec& etemp,const zs::SmallString& PTag,VBufTileVec& vtemp) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        pol(zs::range(vtemp.size()),
            [vtemp = proxy<space>({}, vtemp),PTag] ZS_LAMBDA (int vi) mutable {
                    vtemp(PTag, vi) = (T)0.0;
        });
        pol(zs::range(eles.size()),
                    [vtemp = proxy<space>({},vtemp),etemp = proxy<space>({},etemp),eles = proxy<space>({},eles),HTag,PTag]
                        ZS_LAMBDA(int ei) mutable{
            auto inds = eles.template pack<simplex_dim>("inds",ei).template reinterpret_bits<int>();
            auto H = etemp.template pack<simplex_dim,simplex_dim>(HTag,ei);
            for(int vi = 0;vi != simplex_dim;++vi)
                atomic_add(exec_cuda,&vtemp(PTag,inds[vi]),(T)H(vi,vi));
        });
        pol(zs::range(vtemp.size()),
            [vtemp = proxy<space>({},vtemp),PTag] ZS_LAMBDA(int vi) mutable{
                vtemp(PTag,vi) = 1./vtemp(PTag,vi);
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

        // fmt::print("check point 2\n");
        copy<space_dim * simplex_dim * space_dim * simplex_dim>(pol,elm_buffer,Htag,etemp,"H");
        // fmt::print("check point 3\n");


        multiply<space_dim,simplex_dim>(pol,vtemp,etemp,"H","inds","x","temp");


        // fmt::print("check point 4\n");
        // compute initial residual : b - Hx -> r
        add<space_dim>(pol,vtemp,"b",(T)1.0,"temp",(T)(-1.0),"r");
        project<space_dim>(pol,vtemp,"r","btag");


        // fmt::print("check point 5\n");
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
};