#include "Structures.hpp"
#include "Utils.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"

#include "kernel/geo_math.hpp"

namespace zeno {

struct ZSIsotropicTensionField : INode {
    using dtiles_t = zs::TileVector<float,32>;

    virtual void apply() override {
        using namespace zs;
        auto zssurf = get_input<ZenoParticles>("zssurf");
        auto ref_channel = get_param<std::string>("ref_channel");
        auto def_channel = get_param<std::string>("def_channel");
        auto tension_tag = get_param<std::string>("tension_channel");

        auto& verts = zssurf->getParticles();
        auto& tris = zssurf->getQuadraturePoints();

        if(tris.getPropertySize("inds") != 3) {
            fmt::print("ZSCalcSurfaceTenssionField only supports triangle surface mesh {}\n",tris.getPropertySize("inds"));
            throw std::runtime_error("ZSCalcSurfaceTenssionField only supports triangle surface mesh");
        }
        if(!verts.hasProperty(ref_channel)){
            fmt::print("the input surf does not contain {} channel\n",ref_channel);
            throw std::runtime_error("the input surf does not contain specified referenced channel\n");
        }
        if(!verts.hasProperty(def_channel)) {
            fmt::print("the input surf does not contain {} channel\n",def_channel);
            throw std::runtime_error("the input surf does not contain specified deformed channel\n");
        }

        // if(!verts.hasProperty("nm_incident_facets")) {
        //     // compute the number of incident facets
        //     verts.append_channels({{"nm_incident_facets",1}});
        //     cudaExec(zs::range(verts.size()),
        //         [verts = proxy<space>({},verts)] ZS_LAMBDA(int vi) {
        //             verts("nm_incident_facets",vi) = 0;
        //     });

        //     cudaExec(zs::range(tris.size() * 3),
        //         [verts = proxy<space>({},verts),tris = proxy<space>({},tris)]
        //             ZS_LAMBDA(int ti_dof) mutable {

        //     });
        // }

        // by default we write the facet tension over per facet's specified channel
        auto cudaExec = zs::cuda_exec();
        const auto nmVerts = verts.size();
        const auto nmTris = tris.size();

        if(!verts.hasProperty(tension_tag))
            verts.append_channels(cudaExec,{{tension_tag,1}});
        if(!tris.hasProperty(tension_tag))
            tris.append_channels(cudaExec,{{tension_tag,1}});

        dtiles_t vtemp{verts.get_allocator(),
            {
                {"a",1},
                {"A",1}
            },
        verts.size()};

        dtiles_t etemp{tris.get_allocator(),
            {
                {"A",1},
                {"a",1}
            },
        tris.size()};

        vtemp.resize(verts.size());
        etemp.resize(tris.size());

        constexpr auto space = zs::execspace_e::cuda;



        cudaExec(zs::range(nmTris),
            [tris = proxy<space>({},tris),verts = proxy<space>({},verts),
                    ref_channel = zs::SmallString(ref_channel),
                    def_channel = zs::SmallString(def_channel),
                    etemp = proxy<space>({},etemp)] ZS_LAMBDA (int ti) mutable {
                const auto& inds = tris.template pack<3>("inds",ti).reinterpret_bits<int>(); 
                const auto& X0 = verts.template pack<3>(ref_channel,inds[0]);
                const auto& X1 = verts.template pack<3>(ref_channel,inds[1]);
                const auto& X2 = verts.template pack<3>(ref_channel,inds[2]);
                const auto& x0 = verts.template pack<3>(def_channel,inds[0]);
                const auto& x1 = verts.template pack<3>(def_channel,inds[1]);
                const auto& x2 = verts.template pack<3>(def_channel,inds[2]);
                
                auto A = LSL_GEO::area(X0,X1,X2);
                auto a = LSL_GEO::area(x0,x1,x2);

                etemp("A",ti) = A;
                etemp("a",ti) = a;
                // etemp("ratio",ti) = a / (A + 1e-8);
        });

        cudaExec(zs::range(nmVerts),
            [vtemp = proxy<space>({},vtemp)] ZS_LAMBDA(int vi) mutable {
                vtemp("a",vi) = 0;
                vtemp("A",vi) = 0;
        });

        cudaExec(zs::range(nmTris * 3),
            [tris = proxy<space>({},tris),vtemp = proxy<space>({},vtemp),
                etemp = proxy<space>({},etemp),execTag = wrapv<space>{}] ZS_LAMBDA(int vid) mutable {
            int ti = vid / 3;
            const auto& inds = tris.template pack<3>("inds",ti).reinterpret_bits<int>();
            auto A = etemp("A",ti);
            auto a = etemp("a",ti);
            atomic_add(execTag,&vtemp("A",inds[vid % 3]),A/3);
            atomic_add(execTag,&vtemp("a",inds[vid % 3]),a/3);
        });

        // cudaExec(zs::range(nmVerts),
        //         [vtemp = proxy<space>({},etemp)] ZS_LAMBDA(int tid) mutable {
        //     etemp("A",tid)
        // });

        // blend the tension nodal-wise
        cudaExec(zs::range(nmVerts),
            [verts = proxy<space>({},verts),tension_tag = zs::SmallString(tension_tag),
                    vtemp = proxy<space>({},vtemp)] ZS_LAMBDA (int vi) mutable {
                verts(tension_tag,vi) = vtemp("a",vi) / (vtemp("A",vi) + 1e-8);
        });
        cudaExec(zs::range(nmTris),
            [tris = proxy<space>({},tris),tension_tag = zs::SmallString(tension_tag),
                    etemp = proxy<space>({},etemp)] ZS_LAMBDA (int ti) mutable {
                tris(tension_tag,ti) = etemp("a",ti) / (etemp("A",ti) + 1e-8);
        });        

        set_output("zssurf",zssurf);
    }
};

ZENDEFNODE(ZSIsotropicTensionField, {
                            {"zssurf"},
                            {"zssurf"},
                            {{gParamType_String,"ref_channel","X"},{gParamType_String,"def_channel","x"},{gParamType_String,"tension_channel"," tension"}},
                            {"ZSGeometry"}});

struct ZSEvalAffineTransform : zeno::INode {
    virtual void apply() override {
        using namespace zs;
        auto zsparticles = get_input<ZenoParticles>("zsparticles");
        auto defShapeTag = get_param<std::string>("defShapeTag");
        auto restShapeTag = get_param<std::string>("restShapeTag");
        auto gradientTag = get_param<std::string>("defGradientTag");
        auto transTag = get_param<std::string>("defTransTag");

        const auto& verts = zsparticles->getParticles();
        if(!verts.hasProperty(defShapeTag)) {
            fmt::print("the input zsparticles has no {} channel\n",defShapeTag);
            throw std::runtime_error("the input zsparticles has no specified defShapeTag");
        }
        if(!verts.hasProperty(restShapeTag)) {
            fmt::print("the input zsparticles has no {} channels\n",restShapeTag);
            throw std::runtime_error("the input zsparticles has no specified restShapeTag");
        }

        auto& elms = zsparticles->getQuadraturePoints();
        if(elms.getPropertySize("inds") != 4 && elms.getPropertySize("inds") != 3) {
            fmt::print("the input zsparticles should be a tetrahedra or tri mesh\n");
            throw std::runtime_error("the input zsparticles should be a tetrahedra or tri mesh");
        }

        if(!elms.hasProperty("IB")) {
            fmt::print("the input zsparticles should contain IB channel\n");
            throw std::runtime_error("the input zsparticles should contain IB channel\n"); 
        }

        auto cudaExec = zs::cuda_exec();
        constexpr auto space = zs::execspace_e::cuda;

        if(!elms.hasProperty(gradientTag)) {
            elms.append_channels(cudaExec,{{gradientTag,9}});
        }else if(elms.getPropertySize(gradientTag) != 9) {
            fmt::print("the size of F channel {} is not 9\n",gradientTag);
            throw std::runtime_error("the size of F channel is not 9");
        }
        if(!elms.hasProperty(transTag)) {
            elms.append_channels(cudaExec,{{transTag,3}});
        }else if(elms.getPropertySize(transTag) != 3) {
            fmt::print("the size of b channel {} is not 3\n",transTag);
            throw std::runtime_error("the size of b channel is not 3");
        }

        cudaExec(zs::range(elms.size()),
            [elms = proxy<space>({},elms),verts = proxy<space>({},verts),
                transTag = zs::SmallString(transTag),
                restShapeTag = zs::SmallString(restShapeTag),
                defShapeTag = zs::SmallString(defShapeTag),
                gradientTag = zs::SmallString(gradientTag)] ZS_LAMBDA(int ei) mutable {
            using T = typename RM_CVREF_T(verts)::value_type;
            zs::vec<T,3,3> F{};
            zs::vec<T,3> b{};
            if(elms.propertySize("inds") == 4) {
                const auto& inds = elms.template pack<4>("inds",ei).template reinterpret_bits<int>();
                F = LSL_GEO::deformation_gradient(
                    verts.template pack<3>(defShapeTag,inds[0]),
                    verts.template pack<3>(defShapeTag,inds[1]),
                    verts.template pack<3>(defShapeTag,inds[2]),
                    verts.template pack<3>(defShapeTag,inds[3]),
                    elms.template pack<3,3>("IB",ei));
                auto Xc = zs::vec<T,3>::zeros();
                auto xc = zs::vec<T,3>::zeros();

                for(int i = 0;i != 4;++i){
                    Xc += verts.pack(dim_c<3>,restShapeTag,inds[i])/4.0;
                    xc += verts.pack(dim_c<3>,defShapeTag,inds[i])/4.0;
                }

                b = xc - Xc;
            }else {
                const auto& inds = elms.pack(dim_c<3>,"inds",ei).reinterpret_bits(int_c);
                auto X0 = verts.template pack<3>(restShapeTag,inds[0]);
                auto X1 = verts.template pack<3>(restShapeTag,inds[1]);
                auto X2 = verts.template pack<3>(restShapeTag,inds[2]);

                auto X10 = X1 - X0;
                auto X20 = X2 - X0;
                auto X30 = X10.cross(X20).normalized();
                auto X3 = X30 + X0;

                auto x0 = verts.template pack<3>(defShapeTag,inds[0]);
                auto x1 = verts.template pack<3>(defShapeTag,inds[1]);
                auto x2 = verts.template pack<3>(defShapeTag,inds[2]);

                auto x10 = x1 - x0;
                auto x20 = x2 - x0;
                auto x30 = x10.cross(x20).normalized();
                auto x3 = x30 + x0;         

                LSL_GEO::deformation_xform(X0,X1,X2,X3,x0,x1,x2,x3,F,b);

                // auto tF = LSL_GEO::deformation_gradient(
                //     verts.template pack<3>(defShapeTag,inds[0]),
                //     verts.template pack<3>(defShapeTag,inds[1]),
                //     verts.template pack<3>(defShapeTag,inds[2]),
                //     elms.template pack<2,2>("IB",ei));
                // for(int row = 0;row != 3;++row)
                //     for(int col = 0;col != 2;++col)
                //         F(row,col) = tF(row,col);
                // auto tF0 = col(tF,0);
                // auto tF1 = col(tF,1);
                // auto tF2 = tF0.cross(tF1);
                // tF2 = tF2/(tF2.norm() + (T)1e-6);
                // for(int row = 0;row != 3;++row)
                //     F(row,2) = tF2[row];

                // auto Xc = zs::vec<T,3>::zeros();
                // auto xc = zs::vec<T,3>::zeros();

                // for(int i = 0;i != 3;++i){
                //     Xc += verts.pack(dim_c<3>,restShapeTag,inds[i])/3.0;
                //     xc += verts.pack(dim_c<3>,defShapeTag,inds[i])/3.0;
                // }
                // b = xc - Xc;
            }
            elms.tuple(dim_c<3>,transTag,ei) = b;
            elms.template tuple<9>(gradientTag,ei) = F;
            // if(ei == 0) {
            //     printf("F : \n%f\t%f\t%f\n%f\t%f\t%f\n%f\t%f\t%f\n",
            //         (float)F(0,0),(float)F(0,1),(float)F(0,2),
            //         (float)F(1,0),(float)F(1,1),(float)F(1,2),
            //         (float)F(2,0),(float)F(2,1),(float)F(2,2));
            //     printf("b : %f\t%f\t%f\n",
            //         (float)b[0],(float)b[1],(float)b[2]);
            // }
        });
        // // auto refShapeTag = get_param<std::string>("refShapeTag");

        set_output("zsparticles",zsparticles);
    }
};

ZENDEFNODE(ZSEvalAffineTransform, {
            {"zsparticles"},
            {"zsparticles"},
            {
                {gParamType_String,"defShapeTag","x"},
                {gParamType_String,"defGradientTag","F"},
                {gParamType_String,"restShapeTag","X"},
                {gParamType_String,"defTransTag","b"}
            },
            {"ZSGeometry"}});


struct ZSApplyAffineTransform : zeno::INode {
    virtual void apply() override {
        using namespace zs;
        auto zsparticles = get_input<ZenoParticles>("zsparticles");
        auto restShapeTag = get_param<std::string>("restShapeTag");
        auto defShapeTag = get_param<std::string>("defShapeTag");
        auto gradientTag = get_param<std::string>("defGradientTag");
        auto transTag = get_param<std::string>("defTransTag");
        auto skipTag = get_param<std::string>("skipTag");
        
        auto& verts = zsparticles->getParticles();
        if(!verts.hasProperty(defShapeTag)) {
            fmt::print("the input zsparticles has no {} channel\n",defShapeTag);
            throw std::runtime_error("the input zsparticles has no specified defShapeTag");
        }
        if(!verts.hasProperty(restShapeTag)) {
            fmt::print("the input zsparticles has no {} channels\n",restShapeTag);
            throw std::runtime_error("the input zsparticles has no specified restShapeTag");
        }
        if(!verts.hasProperty(gradientTag)) {
            fmt::print("the input zsparticles has no {} channel\n",gradientTag);
            throw std::runtime_error("the input zsparticles has no nodal-wise deformation gradient");
        }
        if(!verts.hasProperty(transTag)) {
            fmt::print("the input zsparticles has no {} channel\n",transTag);
            throw std::runtime_error("the input zsparticles has no nodal-wise translation");
        }
        if(!verts.hasProperty(skipTag)) {
            fmt::print("the input zsparticles has no {} channel\n",skipTag);
            throw std::runtime_error("the input zsparticles has no nodal-wise skipTag");
        }

        auto cudaExec = zs::cuda_exec();
        constexpr auto space = zs::execspace_e::cuda;

        cudaExec(zs::range(verts.size()),
            [verts = proxy<space>({},verts),
                restShapeTag = zs::SmallString(restShapeTag),
                defShapeTag = zs::SmallString(defShapeTag),
                gradientTag = zs::SmallString(gradientTag),
                transTag = zs::SmallString(transTag),
                skipTag = zs::SmallString(skipTag)] ZS_LAMBDA(int vi) mutable {
            auto X = verts.pack(dim_c<3>,restShapeTag,vi);
            auto F = verts.pack(dim_c<3,3>,gradientTag,vi);
            auto b = verts.pack(dim_c<3>,transTag,vi);

            if(verts(skipTag,vi) > 0.5)
                return;
            verts.tuple(dim_c<3>,defShapeTag,vi) = F * X + b;
            // if(vi == 0){
            //     printf("F : \n%f\t%f\t%f\n%f\t%f\t%f\n%f\t%f\t%f\n",
            //         (float)F(0,0),(float)F(0,1),(float)F(0,2),
            //         (float)F(1,0),(float)F(1,1),(float)F(1,2),
            //         (float)F(2,0),(float)F(2,1),(float)F(2,2));
            // }
            // if(vi == 0) {
            //     auto v = verts.pack(dim_c<3>,restShapeTag,vi);
            //     printf("resulting x : %f %f %f\n",(float)v[0],(float)v[1],(float)v[2]);
            // }
        });

        set_output("zsparticles",zsparticles);
    }
};

ZENDEFNODE(ZSApplyAffineTransform, {
            {"zsparticles"},
            {"zsparticles"},
            {
                {gParamType_String,"skipTag","skipTag"},
                {gParamType_String,"defShapeTag","x"},
                {gParamType_String,"defGradientTag","F"},
                {gParamType_String,"restShapeTag","X"},
                {gParamType_String,"defTransTag","b"}
            },
            {"ZSGeometry"}});

};