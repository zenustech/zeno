#include <zeno/zeno.h>
#include <zeno/logger.h>
#include <zeno/ListObject.h>
#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/utils/UserData.h>
#include <zeno/StringObject.h>


#include <igl/boundary_loop.h>
#include <iostream>
#include <vector>



namespace {
    using namespace zeno;

    struct RetrievePrimitiveBoundaryLoop : zeno::INode {
        virtual void apply() override {
            auto prim = get_input<zeno::PrimitiveObject>("prim");
            size_t nm_vertices = prim->size();
            size_t nm_tris = prim->tris.size();

            Eigen::MatrixXd V;
            Eigen::MatrixXi F;

            V.resize(nm_vertices,3);
            F.resize(nm_tris,3);

            const auto& verts = prim->verts;
            const auto& tris = prim->tris;

            for(int i = 0;i < nm_vertices;++i)
                V.row(i) << verts[i][0],verts[i][1],verts[i][2];
            
            for(int i = 0;i < nm_tris;++i)
                F.row(i) << tris[i][0],tris[i][1],tris[i][2];


            std::vector<std::vector<int>> Lall;
            igl::boundary_loop(F,Lall);   

            size_t nm_loop_vertices = 0;
            for(int i = 0;i < Lall.size();++i)
                nm_loop_vertices += Lall[i].size();
            size_t nm_loop_segs = nm_loop_vertices/* - Lall.size()*/;

            auto loops = std::make_shared<zeno::PrimitiveObject>();

            loops->resize(nm_loop_vertices);
            loops->lines.resize(nm_loop_segs);

            auto& pos = loops->add_attr<zeno::vec3f>("pos");
            auto& IDs = loops->add_attr<float>("ID");
            auto& loopIDs = loops->add_attr<float>("loopID");

            auto& segs = loops->lines;

            int pid = 0;
            int sid = 0;

            for(int i = 0;i < Lall.size();++i) {
                const auto& L = Lall[i];
                int start_pid = pid;
                for(int j = 0;j < L.size();++j) {
                    int id = L[j];

                    pos[pid] = prim->verts[id];
                    IDs[pid] = (float)id;
                    loopIDs[pid] = (float)i;
                    if(j > 0) {
                        segs[sid++] = zeno::vec2i(pid-1,pid);
                    }
                    ++pid;
                }
                int end_pid = pid - 1;
                segs[sid++] = zeno::vec2i(start_pid,end_pid);
            }

            set_output("loops",std::move(loops));
            // set_output2("nm_loops",num)
            // set_ouput("prim",prim);
        }
    };

    ZENDEFNODE(RetrievePrimitiveBoundaryLoop, {
        {"prim"},
        {"loops","prim"},
        {},
        {"Parameterization"},
    });


}