

#include "file_parser/read_vtk_mesh.hpp"
#include "file_parser/write_vtk_unstructured_mesh.hpp"
#include "zensim/omp/execution/ExecutionPolicy.hpp"

#include "zensim/math/bit/Bits.h"
#include "zensim/types/Property.h"
#include <atomic>
#include <zeno/VDBGrid.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>



namespace zeno {

struct ReadVTKMesh : INode {
    void apply() override {
        auto view_interior = get_param<int>("view_interior");
        auto path = get_input<StringObject>("path")->get();
        auto prim = std::make_shared<PrimitiveObject>();
        bool ret = load_vtk_data(path,prim,0);

        if(view_interior && prim->quads.size() > 0){
            prim->tris.resize(prim->quads.size() * 4);
            
            constexpr auto space = zs::execspace_e::openmp;
            auto ompExec = zs::omp_exec();

            ompExec(zs::range(prim->quads.size()),
                [prim] (int ei) mutable {
                    const auto& tet = prim->quads[ei];
                    // for(int i = 0;i < 4;++i)
                    prim->tris[ei * 4 + 0] = zeno::vec3i{tet[0],tet[1],tet[2]};
                    prim->tris[ei * 4 + 1] = zeno::vec3i{tet[1],tet[3],tet[2]};
                    prim->tris[ei * 4 + 2] = zeno::vec3i{tet[0],tet[2],tet[3]};
                    prim->tris[ei * 4 + 3] = zeno::vec3i{tet[0],tet[3],tet[1]};                    
                });
        }

        set_output("prim",std::move(prim));
    }
};

ZENDEFNODE(ReadVTKMesh, {/* inputs: */ {
                            {"readpath", "path"},
                        },
                        /* outputs: */
                        {
                            {"primitive", "prim"},
                        },
                        /* params: */
                        {
                            {"int","view_interior","0"}
                        },
                        /* category: */
                        {
                            "primitive",
                        }});


struct WriteVTKMesh : INode {
    void apply() override {
        auto path = get_input<StringObject>("path")->get();
        auto prim = get_input<PrimitiveObject>("prim");
        int out_customed_nodal = get_param<int>("outVertAttr");
        int out_customed_cell = get_param<int>("outCellAttr");
        bool ret = write_vtk_data(path,prim,out_customed_nodal,out_customed_cell);
        if(!ret){
            throw std::runtime_error("FAILED OUTPUT VTK MESH");
        }

        set_output("prim",std::move(prim));
    }
};

ZENDEFNODE(WriteVTKMesh, {/* inputs: */ {
                            {"primitive", "prim"},
                            {"readpath", "path"},
                        },
                        /* outputs: */
                        {
                            {"primitive", "prim"},
                        },
                        /* params: */
                        {
                            {"int","outVertAttr","0"},
                            {"int","outCellAttr","0"}
                        },
                        /* category: */
                        {
                            "primitive",
                        }});


}

