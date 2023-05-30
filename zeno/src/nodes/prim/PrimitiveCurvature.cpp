#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/UserData.h>
#include <zeno/types/ListObject.h>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

namespace zeno {

namespace {

// 计算点云的曲率
static void computeCurvature(std::shared_ptr<PrimitiveObject> & prim) {
    auto &points = prim->attr<vec3f>("pos");
    if(!prim->verts.has_attr("curvature")){
        prim->verts.add_attr<float>("curvature");
    }
    auto &cur = prim->verts.attr<float>("curvature");
    for (size_t i = 0; i < prim->size(); ++i) {
        auto & p  = points[i];
        auto & p1 = points[i + 1];
        auto & p2 = points[i + 2];
        // 计算曲面法线
        Eigen::Vector3d v1(p1[0] - p[0], p1[1] - p[1], p1[2] - p[2]);
        Eigen::Vector3d v2(p2[0] - p[0], p2[1] - p[1], p2[2] - p[2]);
        Eigen::Vector3d normal = v1.cross(v2).normalized();
        // 构造协方差矩阵
        Eigen::Matrix3d covariance;
        covariance.setZero();
        covariance += v1 * v1.transpose();
        covariance += v2 * v2.transpose();
        // 计算特征值和特征向量
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
        Eigen::Vector3d eigenvalues = solver.eigenvalues();
        // 计算曲率
        cur[i] = eigenvalues.minCoeff() / eigenvalues.sum();
    }
}

// 计算几何体顶点的曲率
static void computeVertexCurvature(std::shared_ptr<PrimitiveObject> & prim) {
    auto &pos = prim->verts;
    if(!prim->verts.has_attr("curvature")){
        prim->verts.add_attr<float>("curvature");
    }
    auto &cur = prim->verts.attr<float>("curvature");
    // 遍历每个顶点
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
// 计算图像的梯度
void computeGradient(std::shared_ptr<PrimitiveObject> & image, std::vector<std::vector<float>>& gradientX, std::vector<std::vector<float>>& gradientY) {
    auto &ud = image->userData();
    int height  = ud.get2<int>("h");
    int width = ud.get2<int>("w");

    gradientX.resize(height, std::vector<float>(width));
    gradientY.resize(height, std::vector<float>(width));

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
struct PrimCurvature: INode {
    void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto type = get_input2<std::string>("type");
        if(type == "object"){
            computeVertexCurvature(prim);
        }
        else if(type == "pointcloud"){
            computeCurvature(prim);
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
ZENDEFNODE(PrimCurvature, {
    {
        {"PrimitiveObject", "prim"},
        {"enum object image pointcloud", "type", "object"},
    },
    {
        {"PrimitiveObject", "prim"},
    },
    {},
    {"primitive"},
});

}
}