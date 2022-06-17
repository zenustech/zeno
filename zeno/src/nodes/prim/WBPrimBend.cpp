#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>


///////////////////////////////////////////////
#include <zeno/utils/orthonormal.h>

#include <math.h>

#include <glm/gtc/epsilon.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/transform.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace zeno {
namespace {

struct WBPrimBend : zeno::INode {
    void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");

        // 是否将变形限制在变形捕获区
        auto limitDeformation = get_input<NumericObject>("Limit Deformation")->get<int>();

        // 变形捕获区外，变形捕获区朝向(Capture Direction)的对侧是否同步变形
        auto symmetricDeformation = get_input<NumericObject>("Symmetric Deformation")->get<int>();

        // 弯曲角度
        auto angle = get_input<zeno::NumericObject>("Bend Angle (degree)")->get<float>();

        // 上向量
        auto upVector = has_input("Up Vector") ? get_input<zeno::NumericObject>("Up Vector")->get<vec3f>() : vec3f(0, 1, 0);

        // 变形捕获区的原点
        auto capOrigin = has_input("Capture Origin") ? get_input<zeno::NumericObject>("Capture Origin")->get<vec3f>() : vec3f(0, 0, 0);

        // 变形捕获区的朝向
        auto dirVector = has_input("Capture Direction") ? get_input<zeno::NumericObject>("Capture Direction")->get<vec3f>() : vec3f(0, 0, 1);

        // 变形捕获区(在 Capture Direction 方向)的长度
        float capLen = has_input("Capture Length") ? get_input<zeno::NumericObject>("Capture Length")->get<float>() : 1.0;

        // 因为不熟悉 zeno 中矩阵的用法，所以转到了 glm
        // 计算变形捕获区的旋转矩阵、逆矩阵
        glm::vec3 up = normalize(glm::vec3(upVector[0],upVector[1],upVector[2]));
        glm::vec3 dir = normalize(glm::vec3(dirVector[0],dirVector[1],dirVector[2]));
        glm::vec3 axis1 = normalize(cross(dir, up));
        glm::vec3 axis2 = cross(axis1, dir);
        glm::vec3 origin = glm::vec3 (capOrigin[0],capOrigin[1],capOrigin[2]);
        float rotMatEle[9] = {dir.x, dir.y, dir.z,
                              axis2.x, axis2.y, axis2.z,
                              axis1.x, axis1.y, axis1.z};
        glm::mat3 rotMat = glm::make_mat3x3(rotMatEle);
        glm::mat3 inverse = glm::transpose(rotMat);

#pragma omp parallel for

        // 开始变形
        for (intptr_t i = 0; i < prim->verts.size(); i++)
        {
            auto pos = prim->verts[i];
            glm::vec3 original = glm::vec3 (pos[0], pos[1], pos[2]);
            glm::vec3 deformedPos = glm::vec3 (pos[0], pos[1], pos[2]);

            // 位置变换到变形捕获区
            original -= origin;
            deformedPos -= origin;

            // Change basis to align capture direction with x axis
            // 因为变形是在变形捕获区中计算的，所以位置和朝向都要变换到变形捕获区
            // 变换后，点的 x 方向就与变形捕获区的朝向(Capture Direction)对齐了
            deformedPos = deformedPos * inverse;
            original = original * inverse;

            float bend_threshold = 0.005;
            if(angle >= bend_threshold)
            {
                // 角度转弧度
                float angleRad  = angle * M_PI / 180;

                // 计算当前 vert 需要变形的幅度
                float bend = angleRad * (deformedPos.x / capLen);

                // Corrected "up" vector is hte positive Y
                glm::vec3 N = { 0, 1, 0 };

                // 计算旋转半径，并将旋转中心点指定到 Y 轴
                glm::vec3 center = (capLen / angleRad) * N;

                // 当前 vert 在 x 方向的位置与变形捕获区长度的比例
                float d = deformedPos.x / capLen;

                // 是否双向同步变形，以及是否将变形限制在变形捕获区
                if (symmetricDeformation)
                {
                    // 如果将变形限制在变形捕获区，超出变形捕获区的长度的 vert 的变形角度按 angleRad 处理
                    if (limitDeformation && abs(deformedPos.x) > capLen)
                    {
                        bend = angleRad;
                        d = 1;
                        if (-deformedPos.x > capLen)
                        {
                            bend *= -1;
                            d *= -1;
                        }
                    }
                }
                else
                {
                    if (deformedPos.x * capLen < 0)
                    {
                        bend = 0;
                        d = 0;
                    }
                    if (limitDeformation && deformedPos.x > capLen)
                    {
                        bend = angleRad;
                        d = 1;
                    }
                }

                // 生成旋转矩阵
                float cb = cos(bend);
                float sb = sin(bend);
                float bendMatEle[9] = {cb, sb, 0,
                                       -sb, cb, 0,
                                       0, 0, 1};
                glm::mat3 bendRotMat = glm::make_mat3x3(bendMatEle);

                // 当前 vert 在 x 方向移回原点
                original.x -= d * capLen;

                // 当前 vert 在 y 方向向下移动的距离为旋转半径
                original -= center;

                // 以原点为圆心，旋转当前 vert
                original = bendRotMat * original;

                // 当前 vert 在 y 方向移回来
                original += center;
            }

            // 从变形捕获区回到世界坐标系
            deformedPos = original * rotMat + origin;

            // 从 glm 转回 zeno
            pos[0] = deformedPos.x;
            pos[1] = deformedPos.y;
            pos[2] = deformedPos.z;

            // 计算结果给到当前 vert
            prim->verts[i] = pos;
        }

        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(WBPrimBend,
           {  /* inputs: */ {
                   {"PrimitiveObject", "prim"},
                   {"int", "Limit Deformation", "1"},
                   {"int", "Symmetric Deformation", "0"},
                   {"float", "Bend Angle (degree)", "90"},
                   {"vec3f", "Up Vector", "0,1,0"},
                   {"vec3f", "Capture Origin", "0,0,0"},
                   {"vec3f", "Capture Direction", "0,0,1"},
                   {"float", "Capture Length", "1.0"},

               }, /* outputs: */ {
                   {"PrimitiveObject", "prim"},
               }, /* params: */ {
               }, /* category: */ {
                   "primitive",
               }});



}
}
