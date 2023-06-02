#include "zeno/zeno.h"
#include "zeno/types/PrimitiveObject.h"
#include "zeno/types/UserData.h"
#include "zeno/types/ListObject.h"
#include <vector>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <igl/point_mesh_squared_distance.h>
#include <igl/gaussian_curvature.h>
#include <igl/principal_curvature.h>

namespace zeno {

namespace {

// 计算图像的梯度
void computeGradient(std::shared_ptr<PrimitiveObject> & image, std::vector<std::vector<float>>& gradientX, std::vector<std::vector<float>>& gradientY) {
    auto &ud = image->userData();
    int height  = ud.get2<int>("h");
    int width = ud.get2<int>("w");

    gradientX.resize(height, std::vector<float>(width));
    gradientY.resize(height, std::vector<float>(width));

#pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (x > 0 && x < width - 1) {
                gradientX[y][x] = (image->verts[y * width + x + 1][0] - image->verts[y * width + x  - 1])[0] / 2.0f;
            } else {
                gradientX[y][x] = 0.0f;
            }
            if (y > 0 && y < height - 1) {
                gradientY[y][x] = (image->verts[(y+1) * width + x][0] - image->verts[(y - 1) * width + x])[0] / 2.0f;
            } else {
                gradientY[y][x] = 0.0f;
            }
        }
    }
}
// 计算图像的曲率
void computeCurvature(std::shared_ptr<PrimitiveObject> & image, const std::vector<std::vector<float>>& gradientX,
                      const std::vector<std::vector<float>>& gradientY) {
    int height = gradientX.size();
    int width = gradientX[0].size();
    if(!image->verts.has_attr("curvature")){
        image->verts.add_attr<float>("curvature");
    }
    auto &cur = image->verts.attr<float>("curvature");
#pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float dx = gradientX[y][x];
            float dy = gradientY[y][x];
            float dxx = 0.0f;
            float dyy = 0.0f;
            float dxy = 0.0f;

            if (x > 0 && x < width - 1) {
                dxx = gradientX[y][x + 1] - 2.0f * dx + gradientX[y][x - 1];
            }

            if (y > 0 && y < height - 1) {
                dyy = gradientY[y + 1][x] - 2.0f * dy + gradientY[y - 1][x];
            }

            if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
                dxy = (gradientX[y + 1][x + 1] - gradientX[y + 1][x - 1] - gradientX[y - 1][x + 1] + gradientX[y - 1][x - 1]) / 4.0f;
            }
            cur[y * width + x] = (dxx * dyy - dxy * dxy) / ((dxx + dyy) * (dxx + dyy) + 1e-6f);
        }
    }
}
// 计算几何体顶点的平均曲率
static void computeVertexCurvature(std::shared_ptr<PrimitiveObject> & prim) {
    auto &cur = prim->verts.add_attr<float>("curvature");
#pragma omp parallel for
    for (size_t i = 0; i < prim->verts.size(); ++i) {
        // 构建顶点的邻域面
        std::vector<size_t> neighborFaces;
        for (int m = 0;m < prim->tris.size();m++) {
            if (prim->tris[m][0] == i || prim->tris[m][1] == i || prim->tris[m][2] == i) {
                neighborFaces.push_back(m);
            }
        }
        // 构建邻域面法线矩阵
        Eigen::MatrixXd normals(3, neighborFaces.size());
        for (size_t j = 0; j < neighborFaces.size(); ++j) {
            auto & face = prim->tris[neighborFaces[j]];
            auto & vert = prim->verts;
            auto & v1 = face[0];
            auto & v2 = face[1];
            auto & v3 = face[2];

            Eigen::Vector3d v12(vert[v2][0] - vert[v1][0], vert[v2][1] - vert[v1][1], vert[v2][2] - vert[v1][2]);
            Eigen::Vector3d v13(vert[v3][0] - vert[v1][0], vert[v3][1] - vert[v1][1], vert[v3][2] - vert[v1][2]);
            Eigen::Vector3d normal = v12.cross(v13).normalized();

            normals(0, j) = normal.x();
            normals(1, j) = normal.y();
            normals(2, j) = normal.z();
        }
        // 计算邻域面法线的协方差矩阵
        Eigen::MatrixXd covariance = (normals * normals.transpose()) / neighborFaces.size();
        // 计算特征值和特征向量
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(covariance);
        Eigen::VectorXd eigenvalues = solver.eigenvalues();
        // 计算曲率
        double curvature = eigenvalues.minCoeff() / eigenvalues.sum();
        cur[i] = curvature;
    }
}

struct PrimCurvature2: INode {
    void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto type = get_input2<std::string>("type");
        if(type == "object"){
            computeVertexCurvature(prim);
        }
        else if(type == "image"){
            auto &ud = prim->userData();
            int w = ud.get2<int>("w");
            int h = ud.get2<int>("h");
            std::vector<std::vector<float>> gx(h, std::vector<float>(w, 0));
            std::vector<std::vector<float>> gy(h, std::vector<float>(w, 0));
            computeGradient(prim,gx, gy);
            computeCurvature(prim,gx,gy);
        }
        set_output("prim", prim);
    }
};
ZENDEFNODE(PrimCurvature2, {
    {
        {"PrimitiveObject", "prim"},
        {"enum object image", "type", "object"},
    },
    {
        {"PrimitiveObject", "prim"},
    },
    {},
    {"deprecated"},
});

//use igl
struct PrimCurvature: INode {
    void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto gaussianCurvature = get_input2<bool>("gaussianCurvature");
        auto curvature = get_input2<bool>("curvature");
        int n = prim->verts.size();
        int dim = 3;
        Eigen::MatrixXd V(n, dim);
        for (int i = 0; i < n; ++i) {
            V.row(i) << prim->verts[i][0], prim->verts[i][1], prim->verts[i][2];
        }
        int m = prim->tris.size();
        int vertices_per_face = 3;
        Eigen::MatrixXi F(m, vertices_per_face);
        for (int i = 0; i < m; ++i) {
            F.row(i) << prim->tris[i][0], prim->tris[i][1], prim->tris[i][2];
        }
        if(gaussianCurvature){
            Eigen::VectorXd K;
            igl::gaussian_curvature(V, F, K);
            prim->verts.add_attr<float>("gaussianCurvature");
            for(int i = 0;i < prim->verts.size();i++){
                prim->verts.attr<float>("gaussianCurvature")[i] = K(i);
            }
        }
        if(curvature){
            Eigen::MatrixXd PD1, PD2;
            Eigen::VectorXd PV1, PV2;
            igl::principal_curvature(V, F, PD1, PD2, PV1, PV2);
            prim->verts.add_attr<float>("curvature");
            for(int i = 0;i < prim->verts.size();i++){
                prim->verts.attr<float>("curvature")[i] = (PV1(i) + PV2(i)) / 2.0;
            }
        }
        set_output("prim", prim);
    }
};
ZENDEFNODE(PrimCurvature, {
    {
        {"PrimitiveObject", "prim"},
        {"bool", "curvature", "0"},
        {"bool", "gaussianCurvature", "0"},
    },
    {
        {"PrimitiveObject", "prim"},
    },
    {},
    {"primitive"},
});

struct MaskByCurvature: INode {
    void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto minCur = get_input2<float>("min_curvature");
        auto maxCur = get_input2<float>("max_curvature");
        auto &mask = prim->verts.attr<float>("mask");
        auto &cur = prim->verts.add_attr<float>("curvature");
#pragma omp parallel for
        for(int i = 0;i < prim->size();i++){
            if (cur[i] > minCur && cur[i] < maxCur) {
                mask[i] = 1;
            }
            else{
                mask[i] = 0;
            }
        }
        set_output("prim", prim);
    }
};
ZENDEFNODE(MaskByCurvature, {
    {
        {"PrimitiveObject", "prim"},
        {"float", "max_curvature", "1"},
        {"float", "min_curvature", "0.005"},
    },
    {
        {"PrimitiveObject", "prim"},

    },
    {},
    {"erode"},
});
}
}