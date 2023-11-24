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

        // Can we do better than element-wise reading?
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
        auto &fnormal = prim->tris.add_attr<zeno::vec3f>("f_normal");
        auto &faces = prim->tris;
        for (int i = 0; i < faces.size(); ++i) {
            zeno::vec3f v1 = normalize(pos[faces[i][1]] - pos[faces[i][0]]);
            zeno::vec3f t = pos[faces[i][2]] - pos[faces[i][0]];
            zeno::vec3f v3 = normalize(cross(v1, t));
            zeno::vec3f v2 = normalize(cross(v1, v3));
            fbx[i] = v1;
            fby[i] = -v2;
            fnormal[i] = v3;
        }

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
        for (auto &it : line_set)
            lines.push_back(zeno::vec2i(it.first, it.second));
        auto meshWhole = new zeno::pmp::SurfaceMesh(prim, "e_feature");

        int N;
        zeno::directional::IntrinsicFaceTangentBundle ftb(meshWhole);
        zeno::directional::CartesianField rawField, combedField;

        bool read_field = read_raw_field("/home/yangkai/Repos/jyz/code/Directional/tutorial/shared/horsers-cf.rawfield", ftb, N, rawField);
        
        //combing and cutting
        rawField.principal_matching();

        // // TODO(@seeeagull): visualize singularities and cuts for debugging
        // for (int i = 0; i < faces.size(); ++i)
        //     for (int j = 0; j < 3; ++j) {
        //         if (face2cut(i, j) == 1) {
        //             clr[faces[i][j]] = clr[faces[i][(j+1)%3]] = vec3f(0.2, 0.3, 0.6);
        //         }
        //     }
        // for (auto &it : rawField.sing_local_cycles)
        //     clr[it] = vec3f(0.5, 0.1, 0.1);

        directional::IntegrationData intData(N);
        Eigen::MatrixXf cut_verts;
        Eigen::MatrixXi cut_faces;
        rawField.setup_integration(intData, cut_verts, cut_faces, combedField);
        auto prim_cut = std::make_shared<PrimitiveObject>();
        auto &pos_cut = prim_cut->attr<vec3f>("pos");
        auto &lines_cut = prim_cut->lines;
        auto &faces_cut = prim_cut->tris;
        pos_cut.resize(cut_verts.rows());
        for (int i = 0; i < cut_verts.rows(); ++i)
            pos_cut[i] = zeno::vec3f(cut_verts(i,0), cut_verts(i,1), cut_verts(i,2));
        faces_cut.resize(cut_faces.rows());
        for (int i = 0; i < cut_faces.rows(); ++i)
            faces_cut[i] = zeno::vec3i(cut_faces(i,0), cut_faces(i,1), cut_faces(i,2));
        std::set<std::pair<int, int>> lines_cut_set{};
        for (auto &it : faces_cut) {
            for (int i = 0; i < 3; ++i) {
                int ii = (i + 1) % 3;
                if (lines_cut_set.count(std::make_pair(it[i], it[ii])) == 0 &&
                    lines_cut_set.count(std::make_pair(it[ii], it[i])) == 0) {
                    lines_cut_set.insert(std::make_pair(it[i], it[ii]));
                }
            }
        }
        lines_cut.clear();
        for (auto &it : lines_cut_set)
            lines_cut.push_back(zeno::vec2i(it.first, it.second));
        auto meshCut = new zeno::pmp::SurfaceMesh(prim_cut, "e_feature");
        
        // Eigen::MatrixXd cutUVFull, cornerWholeUV;
        // intData.verbose=false;
        // intData.integralSeamless = true;
        // intData.roundSeams=false;
        // directional::integrate(combedField,  intData, meshCut, cutUVFull, cornerWholeUV);
        // //Extracting the UV from [U,V,-U, -V];
        // cutUVFull = cutUVFull.block(0,0,cutUVFull.rows(),2);

        prim->tris.erase_attr("fbx");
        prim->tris.erase_attr("fby");
        prim->tris.erase_attr("f_normal");
        // set_output("prim", std::move(prim));
        set_output("prim", std::move(prim_cut));
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
