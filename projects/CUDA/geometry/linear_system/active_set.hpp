#pragma once

#include "mfcg.hpp"

// implement the active set method with bounding box constraints
namespace zeno {
    using T = float;

    template<typename Pol,typename VTileVec>
    void mark_fix_dofs(Pol& pol,VTileVec& vtemp,
        const zs::SmallString& sol_tag,
        const zs::SmallString& ub_tag,
        const zs::SmallString& lb_tag,
        const zs::SmallString& bou_tag,
        const zs::SmallString& fix_tag) {
            using namespace zs;
            constexpr auto space = execspace_e::cuda;
            pol(range(vtemp.size()),
                [vtemp = proxy<space>({},vtemp),sol_tag,ub_tag,lb_tag,bou_tag,fix_tag] __device__(int vi) mutable {
                    auto bou = vtemp(bou_tag,vi);
                    auto x  = vtemp(sol_tag,vi);
                    auto ub = vtemp(ub_tag,vi);
                    auto lb = vtemp(ub_tag,vi);
                    if(bou > 0){ // if it a boundary point, check whether the it exceeds the bounding box
                        if(x < lb || x > ub){
                            printf("boundary vertex<%d> : %f exceeds the bounding box [%f,%f]\n",vi,(float)x,(float)lb,(float)ub);
                            return;
                        }
                        vtemp(fix_tag,vi) = (T)1.0;// fix tag for boundary point
                        return;
                    }
                    // free point
                    if(x < lb){
                        // vtemp(sol_tag,vi) = lb;
                        vtemp(fix_tag,vi) = (T)2.0;// fix tag when exceeding the lower bound
                    }else if(x > ub){
                        // vtemp(sol_tag,vi) = ub;
                        vtemp(fix_tag,vi) = (T)3.0;// fix tag when exceeding the upper bound
                    }else{
                        vtemp(fix_tag,vi) = (T)0.0;
                    }

            });
    }

    template<typename Pol,typename VTileVec>
    void project_constraints(Pol& pol,VTileVec& vtemp,
        const zs::SmallString& sol_tag,
        const zs::SmallString& ub_tag,
        const zs::SmallString& lb_tag,
        const zs::SmallString fix_tag){
            using namespace zs;
            constexpr auto space = execspace_e::cuda;
            pol(range(vtemp.size()),
                [vtemp = proxy<space>({},vtemp),sol_tag,ub_tag,lb_tag,fix_tag] __device__(int vi) mutable {
                    auto fix = vtemp(fix_tag,vi);
                    if(abs(fix - 2.0) < 1e-6){          // exceeding the lower bound, project the dof to lower bound
                        vtemp(sol_tag,vi) = vtemp(lb_tag,vi);
                    }else if(abs(fix - 3.0) < 1e-6){    // exceeding the upper bound, project the dof to upper bound
                        vtemp(sol_tag,vi) = vtemp(ub_tag,vi);
                    }
            });
    }

    // evaluate the lagrangian for 
    template<typename Pol,typename VTileVec,typename ETileVec>
    void update_active_set(Pol& pol,VTileVec& vtemp,ETileVec& etemp,
        const zs::SmallString& H_tag,
        const zs::SmallString& rhs_tag,
        const zs::SmallString& inds_tag,
        const zs::SmallString& sol_tag,
        const zs::SmallString& fix_tag,
        const zs::SmallString& lagrangian_tag,
        T active_set_threshold){
            using namespace zs;
            constexpr auto space = execspace_e::cuda;

            // evaluate the lagrangian l = S(-Ax + b)
            PCG::multiply(pol,vtemp,etemp,H_tag,inds_tag,sol_tag,lagrangian_tag);
            PCG::add(pol,vtemp,lagrangian_tag,(T)-1.0,rhs_tag,(T)1.0,lagrangian_tag);
            cudaPol(range(vtemp.size()),
                [vtemp = proxy<space>({},vtemp),lagrangian_tag,fix_tag,active_set_threshold] __device__(int vi) mutable {
                    auto fix = vtemp(fix_tag,vi);
                    if(fix < 1.0 + 1e-6)
                        vtemp(lagrangian_tag,vi) = 0.0;
                    else if(abs(fix - 3.0) < 1e-6)
                        vtemp(lagrangian_tag,vi) = -vtemp(lagrangian_tag,vi);

                    if(vtemp(lagrangian_tag,vi) < active_set_threshold)
                        vtemp(fix_tag,vi) = 0.0;// set it free
            });
    }

    template<int space_dim,int simplex_dim,typename Pol,typename VTileVec,typename ETileVec>
    int active_set_with_bounding_box_constraints(
        Pol& pol,
        VTileVec& vert_buffer,
        ETileVec& elm_buffer,
        const zs::SmallString& sol_tag,     // solution channel tag
        const zs::SmallString& ub_tag,      // the upper bound channel tag
        const zs::SmallString& lb_tag,      // the lower bound channel tag
        const zs::SmallString& bou_tag,     // fixed boundary value which should not be conflicted with bounding box constriants
        const zs::SmallString& rhs_tag,     // right hand side
        const zs::SmallString& P_tag,       // diagonal preconditioner channel tag
        const zs::SmallString& inds_tag,    // the elm indices' tag
        const zs::SmallString& H_tag,       // hessian tag
        int max_active_set_iters,
        T solution_thresh_hold,
        T cg_rel_accuracy,                  // the relative termination tolerence
        int cg_max_iters,
        int cg_recal_iter = 50
    ){
        static_assert(space_dim == 1,"only scaler field equation is supported");

        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();

        VTileVec vtemp{vert_buffer.get_allocator(),{
            {"sol",1},
            {"sol_old",1},
            {"ub",1},
            {"lb",1},
            {"bou",1},
            {"fix",1},
            {"rhs",1},
            {"P",1},
            {"tmp",1},
            {"lagr",1}  
        },vert_buffer.size()};
        vtemp.resize(vert_buffer.size());

        PCG::copy(pol,vert_buffer,sol_tag,vtemp,"sol");
        PCG::fill(pol,vtemp,"sol_old",vtemp,zs::vec<T,1>{std::numeric_limits<T>::max()});
        PCG::copy(pol,vert_buffer,ub_tag,vtemp,"ub");
        PCG::copy(pol,vert_buffer,lb_tag,vtemp,"lb");
        // bou == 1 means boundary point
        PCG::copy(pol,vert_buffer,bou_tag,vtemp,"bou");
        PCG::copy(pol,vert_buffer,bou_tag,vtemp,"fix");
        PCG::copy(pol,vert_buffer,rhs_tag,vtemp,"rhs");
        PCG::copy(pol,vert_buffer,P_tag,vtemp,"P");

        ETileVec etemp{elm_buffer.get_allocator(),{
            {"inds",simplex_dim},
            {"H",simplex_dim*space_dim*simplex_dim*space_dim}
        },elm_buffer.size()};
        copy<simplex_dim>(pol,elm_buffer,inds_tag,etemp,"inds");
        copy<space_dim * simplex_dim * space_dim * simplex_dim>(pol,elm_buffer,H_tag,etemp,"H");

        int active_set_iters = 0;
        while(active_set_iters < max_active_set_iters){
            // find the breaches of bounding constraints
            mark_fix_dofs(pol,vtemp,"sol","ub","lb","bou","fix");
            PCG::add(pol,vtemp,"sol",(T)1.0,"sol_old",(T)-1.0,"tmp");
            auto diff = std::sqrt(PCG::dot(pol,vtemp,"tmp","tmp"));
            if(diff < solution_thresh_hold)
                break;
            PCG::copy(pol,vtemp,"sol",vtemp,"sol_old");
            project_constraints(pol,vtemp,"sol","ub","lb","fix");

            pcg_with_fixed_sol_solve(pol,vtemp,etemp,"sol","fix","rhs","P","inds","H",cg_rel_accuracy,cg_max_iters,cg_recal_iter);
            // remove the inactive constraints from active set
            update_active_set(pol,vtemp,etemp,"H","rhs","inds","sol","fix","lagr");
            ++active_set_iters;
        }
    }
};