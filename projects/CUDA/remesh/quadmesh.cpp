#include <cctype>
#include <filesystem>
#include <sstream>
#include <fstream>
#include <zeno/core/Graph.h>
#include <zeno/extra/assetDir.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/utils/log.h>
#include <zeno/zeno.h>
#include <Eigen/Core>
#include "./directional/CartesianField.h"
#include "./directional/IntrinsicFaceTangentBundle.h"


namespace zeno {

bool read_raw_field(const std::string &fileName,
                    const zeno::directional::IntrinsicFaceTangentBundle& tb,
                    int& N,
                    zeno::directional::CartesianField& field) {
    try {
        std::ifstream f(fileName);
        if (!f.is_open()) {
            return false;
        }
        int numT;
        f>>N;
        f>>numT;
        Eigen::MatrixXf extField;
        extField.conservativeResize(numT, 3*N);

        //Can we do better than element-wise reading?
        for (int i=0;i<extField.rows();i++)
            for (int j=0;j<extField.cols();j++)
                f>>extField(i,j);

        f.close();
        assert(tb.sources.rows()==extField.rows());
        assert(tb.hasEmbedding() && "This tangent bundle doesn't admit an extrinsic embedding");
        field.init(tb, zeno::directional::fieldTypeEnum::RAW_FIELD, N);
        field.set_extrinsic_field(extField);
        return f.good();
    }
    catch (std::exception e) {
        return false;
    }
}

struct QuadMeshing : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto &pos = prim->attr<vec3f>("pos");
        auto &clr = prim->verts.add_attr<vec3f>("clr");
        auto &lines = prim->lines;
        auto &fbx = prim->tris.add_attr<zeno::vec3f>("fbx");
        auto &fby = prim->tris.add_attr<zeno::vec3f>("fby");
        auto &faces = prim->tris;
        for (int i = 0; i < faces.size(); ++i) {
            zeno::vec3f v1 = normalize(pos[faces[i][1]] - pos[faces[i][0]]);
            zeno::vec3f t = pos[faces[i][2]] - pos[faces[i][0]];
            zeno::vec3f v3 = normalize(cross(v1, t));
            zeno::vec3f v2 = normalize(cross(v1, v3));

            fbx[i] = v1;
            fby[i] = -v2;
            // normal = v3;
        }

        zeno::log_info("quad meshing start");
        std::set<std::pair<int, int>> line_set{};
        for (auto &it : prim->tris) {
            for (int i = 0; i < 3; ++i) {
                int ii = (i + 1) % 3;
                if (line_set.count(std::make_pair(it[i], it[ii])) == 0 &&
                    line_set.count(std::make_pair(it[ii], it[i])) == 0) {
                    line_set.insert(std::make_pair(it[i], it[ii]));
                }
            }
        }
        lines.clear();
        for (auto &it : line_set) {
            lines.push_back(zeno::vec2i(it.first, it.second));
        }
        auto meshWhole = new zeno::pmp::SurfaceMesh(prim, "e_feature");
        zeno::log_info("build surface mesh");

        int N;
        zeno::directional::IntrinsicFaceTangentBundle ftb(meshWhole);
        zeno::log_info("init tangent bundle");
        zeno::directional::CartesianField rawField, combedField;

        bool read_field = read_raw_field("/home/yangkai/Repos/jyz/code/Directional/tutorial/shared/horsers-cf.rawfield", ftb, N, rawField);
        zeno::log_info("read raw field: {}", read_field);
        
        //combing and cutting
        rawField.principal_matching();
        zeno::log_info("principal matching done");
        Eigen::MatrixXi face2cut =  Eigen::MatrixXi::Zero(faces.size(), 3);
        rawField.cut_mesh_with_singularities(rawField.sing_local_cycles, face2cut);
        zeno::log_info("cut mesh with singularities done");
        rawField.combing(combedField, face2cut);
        zeno::log_info("combing done");

        // TODO(@seeeagull): visualize singularities and cuts for debugging
        for (int i = 0; i < faces.size(); ++i) {
            for (int j = 0; j < 3; ++j) {
                if (face2cut(i, j) == 1) {
                    clr[faces[i][j]] = clr[faces[i][(j+1)%3]] = vec3f(0.2, 0.3, 0.6);
                }
            }
        }
        for (auto &it : rawField.sing_local_cycles) {
            clr[it] = vec3f(0.5, 0.1, 0.1);
        }

        // Eigen::MatrixXd cutUVFull, cornerWholeUV;

        // directional::IntegrationData intData(N);
        // directional::setup_integration(rawField, intData, meshCut, combedField);
        
        // intData.verbose=false;
        // intData.integralSeamless = true;
        // intData.roundSeams=false;
        // directional::integrate(combedField,  intData, meshCut, cutUVFull, cornerWholeUV);
        // //Extracting the UV from [U,V,-U, -V];
        // cutUVFull = cutUVFull.block(0,0,cutUVFull.rows(),2);

        prim->tris.erase_attr("fbx");
        prim->tris.erase_attr("fby");
        set_output("prim", std::move(prim));
    }
};

ZENO_DEFNODE(QuadMeshing)
({
    {{"prim"}},
    {"prim"},
    {},
    {"primitive"},
});


} // namespace zeno
