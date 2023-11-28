#include <cctype>
#include <filesystem>
#include <sstream>
#include <zeno/core/Graph.h>
#include <zeno/extra/assetDir.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/utils/log.h>
#include <zeno/zeno.h>
#include <Eigen/Core>


namespace zeno {


struct UniformRemeshing : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto &pos = prim->attr<vec3f>("pos");

        auto meshWhole = new zeno::pmp::SurfaceMesh(prim, line_pick_tag);

        int N;
        directional::TriMesh meshWhole, meshCut;
        directional::IntrinsicFaceTangentBundle ftb;
        directional::CartesianField rawField, combedField;
        Eigen::MatrixXd cutUVFull, cutUVRot, cornerWholeUV;

        directional::readOFF(TUTORIAL_SHARED_PATH "/horsers.off", meshWhole);
        ftb.init(meshWhole);
        directional::read_raw_field(TUTORIAL_SHARED_PATH "/horsers-cf.rawfield", ftb, N, rawField);
        
        //combing and cutting
        directional::principal_matching(rawField);
        directional::IntegrationData intData(N);
        directional::setup_integration(rawField, intData, meshCut, combedField);
        
        intData.verbose=false;
        intData.integralSeamless = true;
        intData.roundSeams=false;
        directional::integrate(combedField,  intData, meshCut, cutUVFull, cornerWholeUV);
        //Extracting the UV from [U,V,-U, -V];
        cutUVFull=cutUVFull.block(0,0,cutUVFull.rows(),2);

        set_output("prim", std::move(prim));
    }
};

ZENO_DEFNODE(UniformRemeshing)
({
    {{"prim"}},
    {"prim"},
    {},
    {"primitive"},
});


} // namespace zeno
