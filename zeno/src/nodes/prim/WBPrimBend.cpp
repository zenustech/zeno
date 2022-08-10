//
// Created by WangBo on 2022/7/5.
//

#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/MatrixObject.h>
#include <zeno/utils/orthonormal.h>
#include <zeno/utils/random.h>
#include <cmath>
#include <glm/gtx/quaternion.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace zeno
{
namespace
{
// houdini 风格的 bend 变形器
struct WBPrimBend : INode {
    void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");

        // 是否将变形限制在变形捕获区
        auto limitDeformation = get_input<NumericObject>("Limit Deformation")->get<int>();

        // 变形捕获区外，变形捕获区朝向(Capture Direction)的对侧是否同步变形
        auto symmetricDeformation = get_input<NumericObject>("Symmetric Deformation")->get<int>();

        // 弯曲角度
        auto angle = get_input<NumericObject>("Bend Angle (degree)")->get<float>();

        // 上向量
        auto upVector = has_input("Up Vector") ? get_input<NumericObject>("Up Vector")->get<vec3f>() : vec3f(0, 1, 0);

        // 变形捕获区的原点
        auto capOrigin = has_input("Capture Origin") ? get_input<NumericObject>("Capture Origin")->get<vec3f>() : vec3f(0, 0, 0);

        // 变形捕获区的朝向
        auto dirVector = has_input("Capture Direction") ? get_input<NumericObject>("Capture Direction")->get<vec3f>() : vec3f(0, 0, 1);

        // 变形捕获区(在 Capture Direction 方向)的长度
        double capLen = has_input("Capture Length") ? get_input<NumericObject>("Capture Length")->get<float>() : 1.0;

        // 因为不熟悉 zeno 中矩阵的用法，所以转到了 glm
        // 计算变形捕获区的旋转矩阵、逆矩阵
        glm::vec3 up = normalize(glm::vec3(upVector[0],upVector[1],upVector[2]));
        glm::vec3 dir = normalize(glm::vec3(dirVector[0],dirVector[1],dirVector[2]));
        glm::vec3 axis1 = normalize(cross(dir, up));
        glm::vec3 axis2 = cross(axis1, dir);
        glm::vec3 origin = glm::vec3 (capOrigin[0],capOrigin[1],capOrigin[2]);
        double rotMatEle[9] = {dir.x, dir.y, dir.z,
                              axis2.x, axis2.y, axis2.z,
                              axis1.x, axis1.y, axis1.z};
        glm::mat3 rotMat = glm::make_mat3x3(rotMatEle);
        glm::mat3 inverse = glm::transpose(rotMat);

#pragma omp parallel for
        for (intptr_t i = 0; i < prim->size(); i++)
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
            deformedPos = inverse * deformedPos;
            original = inverse * original;

            double bend_threshold = 0.005;
            if(std::abs(angle) >= bend_threshold)
            {
                // 角度转弧度
                double angleRad  = angle * M_PI / 180;

                // 计算当前 vert 需要变形的幅度
                double bend = angleRad * (deformedPos.x / capLen);

                // Corrected "up" vector is hte positive Y
                glm::vec3 N = { 0, 1, 0 };

                // 计算旋转半径，并将旋转中心点指定到 Y 轴
                glm::vec3 center = (float)(capLen / angleRad) * N;

                // 当前 vert 在 x 方向的位置与变形捕获区长度的比例
                double d = deformedPos.x / capLen;

                // 是否双向同步变形，以及是否将变形限制在变形捕获区
                if (symmetricDeformation)
                {
                    // 如果将变形限制在变形捕获区，超出变形捕获区的长度的 vert 的变形角度按 angleRad 处理
                    if (limitDeformation && std::abs(deformedPos.x) > capLen)
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
                double cb = std::cos(bend);
                double sb = std::sin(bend);
                double bendMatEle[9] = {cb, sb, 0,
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
            deformedPos = rotMat * original + origin;

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


// 打印节点信息，调试用
struct PrintPrimInfo : INode {
    void apply() override
    {
        // 额外的标记
        int primOutPutType = 0; // 0 = ref, 1 = copy
        int originalPrimModified = 0;

        auto prim = get_input<PrimitiveObject>("prim");

        auto vertCount = std::make_shared<NumericObject>();
        auto pointCount = std::make_shared<NumericObject>();
        auto lineCount = std::make_shared<NumericObject>();
        auto triCount = std::make_shared<NumericObject>();
        auto quadCount = std::make_shared<NumericObject>();
        vertCount->set<int>(int(prim->size()));
        pointCount->set<int>(int(prim->points.size()));
        lineCount->set<int>(int(prim->lines.size()));
        triCount->set<int>(int(prim->tris.size()));
        quadCount->set<int>(int(prim->quads.size()));

        if (get_param<bool>("printInfo"))
        {
            std::vector<std::string> myKeys = prim->attr_keys();
            printf("--------------------------------\n");
            printf("wb-Debug ==> vert has attr:\n");
            for(const std::string& i : myKeys)
            {

                printf("wb-Debug ==> vertAttr.%s\n", i.c_str());
            }
            printf("--------------------------------\n");
            printf("wb-Debug ==> vertsCount = %i\n", vertCount->get<int>());
            printf("wb-Debug ==> pointsCount = %i\n", pointCount->get<int>());
            printf("wb-Debug ==> linesCount = %i\n", lineCount->get<int>());
            printf("wb-Debug ==> trisCount = %i\n", triCount->get<int>());
            printf("wb-Debug ==> quadsCount = %i\n", quadCount->get<int>());
            printf("--------------------------------\n");
        }

        set_output("prim", std::move(prim));
    }
};
ZENDEFNODE(PrintPrimInfo,
           {  /* inputs: */ {
                   "prim",
               }, /* outputs: */ {
                   "prim",
               }, /* params: */ {
                   {"int", "printInfo", "1"},
               }, /* category: */ {
                   "primitive",
               }});


// 指定段数和半径的圆片
struct CreateCircle : INode {
    void apply() override
    {
        auto seg = get_input<NumericObject>("segments")->get<int>();
        auto r = get_input<NumericObject>("r")->get<float>();
        auto prim = std::make_shared<PrimitiveObject>();

        for (int i = 0; i < seg; i++)
        {
            float rad = 2.0 * M_PI * i / seg;
            prim->verts.push_back(vec3f(cos(rad)*r, 0, -sin(rad)*r));
        }
        prim->verts.push_back(vec3i(0, 0, 0));

        for (int i = 0; i < seg; i++)
        {
            prim->tris.push_back(vec3i(seg, i, (i + 1) % seg));
        }

        set_output("prim", std::move(prim));
    }
};
ZENDEFNODE(CreateCircle,
           {  /* inputs: */ {
                   {"int","segments","32"},
                   {"float","r","1.0"},
               }, /* outputs: */ {
                   "prim",
               }, /* params: */ {
               }, /* category: */ {
                   "primitive",
               }});


// 指定起点朝向和结束朝向，计算旋转四元数
struct QuatRotBetweenVectors : INode {
    void apply() override
    {
        auto start = get_input<NumericObject>("start")->get<vec3f>();
        auto dest = get_input<NumericObject>("dest")->get<vec3f>();

        glm::vec3 gl_start(start[0], start[1], start[2]);
        glm::vec3 gl_dest(dest[0], dest[1], dest[2]);

        vec4f rot;
        glm::quat quat = glm::rotation(gl_start, gl_dest);
        rot[0] = quat.x;
        rot[1] = quat.y;
        rot[2] = quat.z;
        rot[3] = quat.w;

        auto rotation = std::make_shared<NumericObject>();
        rotation->set<vec4f>(rot);
        set_output("quatRotation", rotation);
    }
};
ZENDEFNODE(QuatRotBetweenVectors,
           {  /* inputs: */ {
                   {"vec3f", "start", "1,0,0"},
                   {"vec3f", "dest", "1,0,0"},
               }, /* outputs: */ {
                   {"vec4f", "quatRotation", "0,0,0,1"},
               }, /* params: */ {
               }, /* category: */ {
                   "math",
               }});


// 矩阵转置
struct MatTranspose : INode{
    void apply() override
    {
        glm::mat mat = std::get<glm::mat4>(get_input<MatrixObject>("mat")->m);
        glm::mat transposeMat = glm::transpose(mat);
        auto oMat = std::make_shared<MatrixObject>();
        oMat->m = transposeMat;
        set_output("transposeMat", oMat);
    }
};
ZENDEFNODE(MatTranspose,
           { /* inputs: */ {
                   "mat",
               }, /* outputs: */ {
                   "transposeMat",
               }, /* params: */ {
               }, /* category: */ {
                   "math",
               } });


// 线上重新采样
struct LineResample : INode {
    void apply() override
    {
        // 额外的标记
        int primOutPutType = 1; // 0 = ref, 1 = copy
        int originalPrimModified = 0;

        auto prim = get_input<PrimitiveObject>("prim");
        auto segments = get_input<NumericObject>("segments")->get<int>();
        if (segments < 1) { segments = 1; }

        float total = 0;
        std::vector<float> linesLen(prim->lines.size());   // u 记录线的累加长度
        for (size_t i = 0; i < prim->lines.size(); i++) {
            auto const &ind = prim->lines[i];
            auto a = prim->verts[ind[0]];
            auto b = prim->verts[ind[1]];
            auto area = length(b - a);
            total += area;
            linesLen[i] = total;
        }
        auto inv_total = 1 / total;
        for (auto &_linesLen : linesLen) { _linesLen *= inv_total; } // 取值范围映射到 0.0 ~ 1.0

        auto retprim = std::make_shared<PrimitiveObject>(); // 构建一个新的 prim
        retprim->resize(segments + 1);
        retprim->add_attr<float>("curveU");
        for (int idx = 0; idx <= segments; idx++)
        {
            float delta = 1.0f / float(segments);
            float insertU = float(idx) * delta;
            if (idx == segments) { insertU = 1; }

            // 在指定区域内用二分法查找大于目标值的最小元素
            auto it = std::lower_bound(linesLen.begin(), linesLen.end(), insertU);
            size_t index = it - linesLen.begin();
            index = std::min(index, prim->lines.size() - 1);

            auto const &ind = prim->lines[index];   // 存放着这个线的起点和终点
            auto a = prim->verts[ind[0]];   // ind[0] = 1，起点的索引号
            auto b = prim->verts[ind[1]];   // ind[1] = 2，终点的索引号

            // 根据 insertU 来计算新插入的点是位置
            auto r1 = (insertU - linesLen[index - 1])/(linesLen[index] - linesLen[index - 1]);   // 插值系数
            auto p = a + (b - a) * r1;

            retprim->verts[idx] = p;    // 设置新 prim 中点的位置
            auto &cu = retprim->attr<float>("curveU");
            cu[idx] = insertU;  // 设置新 prim 的属性
        }

        // 为新 prim 构建 lines
        retprim->lines.reserve(retprim->size());
        for (int i = 1; i < retprim->size(); i++) {
            retprim->lines.emplace_back(i - 1, i);
        }

        set_output("prim", std::move(retprim));
    }
};
ZENDEFNODE(LineResample,
           {  /* inputs: */ {
                   "prim",
                   {"int", "segments", "3"},
               }, /* outputs: */ {
                   "prim",
               }, /* params: */ {
               }, /* category: */ {
                   "primitive",
               }});


// 裁剪线条
struct LineCarve : INode {
    void apply() override
    {
        // 额外的标记
        int primOutPutType = 0; // 0 = ref, 1 = copy
        int originalPrimModified = 1;

        auto prim = get_input<PrimitiveObject>("prim");
        auto insertU = get_input<NumericObject>("insertU")->get<float>();

        float total = 0;
        std::vector<float> linesLen(prim->lines.size());   // u 记录线的累加长度
        for (size_t i = 0; i < prim->lines.size(); i++) {
            auto const &ind = prim->lines[i];
            auto a = prim->verts[ind[0]];
            auto b = prim->verts[ind[1]];
            auto area = length(b - a);
            total += area;
            linesLen[i] = total;
        }
        auto inv_total = 1 / total;
        for (auto &_linesLen : linesLen) { _linesLen *= inv_total; } // 取值范围映射到 0.0 ~ 1.0

        // 在指定区域内用二分法查找大于目标值的最小元素
        insertU = std::min(1.f, std::max(0.f, insertU));
        auto it = std::lower_bound(linesLen.begin(), linesLen.end(), insertU);
        size_t index = it - linesLen.begin();
//        printf("wb-Debug ==> 01 line_index = %i\n", int(index));
        index = std::min(index, prim->lines.size() - 1);
        if (index < 0) index = 0;

        auto const &ind = prim->lines[index];   // 存放着这个线的起点和终点
        auto a = prim->verts[ind[0]];   // ind[0] = 1，起点的索引号
        auto b = prim->verts[ind[1]];   // ind[1] = 2，终点的索引号

        // 根据 insertU 来计算新插入的点是位置
        auto r1 = (insertU - linesLen[index - 1])/(linesLen[index] - linesLen[index - 1]);   // 插值系数
        auto p = a + (b - a) * r1;

        // 往 verts 里面插入新的 vert
        // prim->verts.resize(n); 不应该用这个，这个会将游标移到末端
        prim->verts.reserve(prim->size() + 1);  // 容量 + 1，但游标没有移到末端
        prim->attr<vec3f>("pos").insert(    // 插入元素对象
            prim->attr<vec3f>("pos").begin() + int(index) + 1, {p[0], p[1], p[2]});

        // 更新线的长度 vector linesLen，线的数量也要增加一个
        linesLen.reserve(linesLen.size() + 1);    // u.size() = 10
        linesLen.insert(linesLen.begin() + int(index), insertU);   // u.size() = 11

        // 根据 vector linesLen 更新 verts 的 curveU 属性
        if (!prim->has_attr("curveU")) {
            prim->add_attr<float>("curveU");
        }
        auto &cu = prim->verts.attr<float>("curveU");
        cu[0] = 0.0f;
        for (size_t i = 1; i < prim->size(); i++) {
            cu[i] = linesLen[i - 1];
        }

        if (get_input<NumericObject>("cut")->get<int>()) // 切掉一部分点
        {
            //if (get_param<bool>("cut insert to end"))    // 切掉后面部分
            if (get_input<NumericObject>("cut insert to end")->get<int>())
            {
                if (prim->verts.begin() + int(index) <= prim->verts.end())
                {
//                    printf("wb-Debug ==> 02 line_index = %i\n", int(index));
                    prim->verts->erase(prim->verts.begin() + int(index) + 2, prim->verts.end());
                    cu.erase(cu.begin() + int(index) + 2, cu.end());
                }
            }
            else
            {
                if (prim->verts.begin() + int(index) <= prim->verts.end())
                {
//                    printf("wb-Debug ==> 02 line_index = %i\n", int(index));
                    prim->verts->erase(prim->verts.begin(), prim->verts.begin() + int(index) + 1);
                    cu.erase(cu.begin(), cu.begin() + int(index) + 1);
                }
            }
        }

        // 重建 lines
        prim->lines.clear();
        size_t line_n = prim->verts.size();
        prim->lines.reserve(line_n);
        for (size_t i = 1; i < line_n; i++) {
            prim->lines.emplace_back(i - 1, i);
        }
        prim->lines.update();

        set_output("prim", std::move(prim));
    }
};
ZENDEFNODE(LineCarve,
           {  /* inputs: */ {
                   "prim",
                   {"float", "insertU", "0.15"},
                   {"bool", "cut", "0"},
                   {"bool", "cut insert to end", "1"},
               }, /* outputs: */ {
                   "prim",
               }, /* params: */ {

               }, /* category: */ {
                   "primitive",
               }});


// vec3 属性数据可视化
struct VisVec3Attribute : INode {
    void apply() override
    {
        // 额外的标记
        int primOutPutType = 1; // 0 = ref, 1 = copy
        int originalPrimModified = 0;

        auto color = get_input<NumericObject>("color")->get<vec3f>();
        auto useNormalize = get_input<NumericObject>("normalize")->get<int>();
        auto lengthScale = get_input<NumericObject>("lengthScale")->get<float>();
        auto name = get_input2<std::string>("name");
        auto prim = get_input<PrimitiveObject>("prim");

        auto &attr = prim->attr<vec3f>(name);

        // 构建新的可视化的 Prim
        auto primVis = std::make_shared<PrimitiveObject>();
        primVis->verts.resize(prim->size() * 2);
        primVis->lines.resize(prim->size());

        auto &visColor = primVis->add_attr<vec3f>("clr");

#pragma omp parallel for
        for (int iPrim = 0; iPrim < prim->size(); iPrim++)
        {
            int i = iPrim * 2;
            primVis->verts[i] = prim->verts[iPrim];
            visColor[i] = color;
            ++i; // primVis
            auto a=attr[iPrim];
            if (useNormalize) a = normalize(a);
            primVis->verts[i] = prim->verts[iPrim] + a * lengthScale;
            visColor[i] = color * 0.25;
            primVis->lines[i/2][0] = i-1;  // 线
            primVis->lines[i/2][1] = i;  // 线
        }

        set_output("primVis", std::move(primVis));
    }
};
ZENDEFNODE(VisVec3Attribute,
           {  /* inputs: */ {
                   "prim",
                   {"string", "name", "vel"},
                   {"bool", "normalize", "1"},
                   {"float", "lengthScale", "1.0"},
                   {"vec3f", "color", "1,1,0"},
               }, /* outputs: */ {
                   "primVis",
               }, /* params: */ {
               }, /* category: */ {
                   "visualize",
               }});


// 相当于 houdini 中的 trail Node + add Node
struct TracePositionOneStep : INode {
    void apply() override
    {
        // 额外的标记
        int primOutPutType = 1;         // 0 = ref, 1 = copy
        int originalPrimModified = 1;   // primEmpty

        // pos 被数据修改后的 prim
        auto primData = get_input<PrimitiveObject>("primData");

        // 用于可视化的 primVis
        auto primVis = get_input<PrimitiveObject>("primStart");

        // 用于标记线的 id，便于将来将每条线分出来
        auto idName = get_input2<std::string>("lineTag");
        if (!primVis->has_attr(idName)) { primVis->add_attr<int>(idName); }

        // OneStep 之前点和线的数量
        int primVisVertsCount = (int)primVis->size();
        int primVislinesCount = (int)primVis->lines.size();
        int primDataVertsCount = (int)primData->size();;

        // 有可能 primEmpty 会自带一些点进来，这些点也需要分配 lineID 属性
        if(primVisVertsCount != 0)
        {
            if(primVislinesCount == 0)
            {
                for (int i = 0; i < primVisVertsCount; i++)
                {
                    primVis->attr<int>(idName)[i] = i;
                }
            }
            // 有了基础点才需要构造线
            primVis->lines.resize(primVislinesCount + primDataVertsCount);
        }

        // 为填入新的点和线准备空间
        primVis->verts.resize(primVisVertsCount + primDataVertsCount);

#pragma omp parallel for
        for (int i = 0; i < primDataVertsCount; i++)
        {
            // 新点的位置 = primIn 上点的位置
            primVis->verts[primVisVertsCount + i] = primData->verts[i];

            // 线的序号标记
            primVis->attr<int>(idName)[primVisVertsCount + i] = i;

            // 如果这是第一轮输入的点，就不用构造线
            if(primVisVertsCount != 0)
            {
                // 新线的索引 = 上一轮新点的索引 : 本轮新点的索引
                primVis->lines[primVislinesCount + i][0] = primVisVertsCount + i - primDataVertsCount;
                primVis->lines[primVislinesCount + i][1] = primVisVertsCount + i;
            }
        }

        set_output("primVis", std::move(primVis));
    }
};
ZENDEFNODE(TracePositionOneStep,
           {  /* inputs: */ {
                   "primData",
                   "primStart",
                   {"string", "lineTag", "lineID"},
               }, /* outputs: */ {
                   "primVis",
               }, /* params: */ {
               }, /* category: */ {
                   "visualize",
               }});


struct PrimAddVec3fAttr : INode {
    void apply() override
    {
        int primOutPutType = 0; // 0 = ref, 1 = copy
        int originalPrimModified = 1;

        auto prim = get_input<PrimitiveObject>("prim");
        auto name = get_input2<std::string>("name");
        auto value = get_input<NumericObject>("value")->get<vec3f>();
        if (!prim->has_attr(name))
        {
            prim->add_attr<vec3f>(name, value);
        }
        else
        {
            auto &attr_arr = prim->attr<vec3f>(name);
#pragma omp parallel for
            for (intptr_t i = 0; i < prim->size(); i++)
            {
                attr_arr[i] = value;
            }
        }

        set_output("prim", std::move(prim));
    }
};
ZENDEFNODE(PrimAddVec3fAttr,
           { /* inputs: */ {
                   "prim",
                   {"string", "name", "pos"},
                   {"vec3f", "value", "0,0,0"},
               }, /* outputs: */ {
                   "prim",
               }, /* params: */ {
               }, /* category: */ {
                   "primitive",
               } });


struct PrimAddFloatAttr : INode {
    void apply() override
    {
        int primOutPutType = 0; // 0 = ref, 1 = copy
        int originalPrimModified = 1;

        auto prim = get_input<PrimitiveObject>("prim");
        auto name = get_input2<std::string>("name");
        auto value = get_input<NumericObject>("value")->get<float>();

        if (!prim->has_attr(name))
        {
            prim->add_attr<float>(name, value);
        }
        else
        {
            auto &attr_arr = prim->attr<float>(name);
#pragma omp parallel for
            for (intptr_t i = 0; i < prim->size(); i++)
            {
                attr_arr[i] = value;
            }
        }

        set_output("prim", std::move(prim));
    }
};
ZENDEFNODE(PrimAddFloatAttr,
           { /* inputs: */ {
                   "prim",
                   {"string", "name", "pos"},
                   {"float", "value", "0"},
               }, /* outputs: */ {
                   "prim",
               }, /* params: */ {
               }, /* category: */ {
                   "primitive",
               } });


// 根据属性名称设置每点属性，name 是 input
struct PrimSetAttrPerVertByName : INode {
    void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto name = get_input2<std::string>("name");
        auto type = get_input2<std::string>("type");
        auto index = get_input<NumericObject>("index")->get<int>();

        if (type == "float") {
            auto value = get_input<NumericObject>("value")->get<float>();
            auto &attr_arr = prim->attr<float>(name);
            if (index < attr_arr.size()) {
                attr_arr[index] = value;
            }
        }
        else if (type == "float3") {
            auto value = get_input<NumericObject>("value")->get<vec3f>();
            auto &attr_arr = prim->attr<vec3f>(name);
            if (index < attr_arr.size()) {
                attr_arr[index] = value;
            }
        }
        else {
            throw Exception("Bad attribute type: " + type);
        }
    }
};
ZENDEFNODE(PrimSetAttrPerVertByName,
           { /* inputs: */ {
                   "prim",
                   {"string", "name", "pos"},
                   {"enum float float3", "type", "float3"},
                   {"int","index","0"},
                   "value",
               }, /* outputs: */ {
               }, /* params: */ {
               }, /* category: */ {
                   "primitive",
               } });


// 根据属性名称获取每点属性，name 是 input
struct PrimGetAttrPerVertByName : INode {
    void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto name = get_input2<std::string>("name");
        auto type = get_input2<std::string>("type");
        auto index = get_input<NumericObject>("index")->get<int>();
        auto value = std::make_shared<NumericObject>();

        if (type == "float") {
            value->set<float>(0);
            auto &attr_arr = prim->attr<float>(name);
            if (index < attr_arr.size()) {
                value->set<float>(attr_arr[index]);
            }
        }
        else if (type == "float3") {
            value->set<vec3f>(vec3f(0, 0, 0));
            auto &attr_arr = prim->attr<vec3f>(name);
            if (index < attr_arr.size()) {
                value->set<vec3f>(attr_arr[index]);
            }
        }
        else {
            throw Exception("Bad attribute type: " + type);
        }

        set_output("value", std::move(value));
    }
};
ZENDEFNODE(PrimGetAttrPerVertByName,
           { /* inputs: */ {
                   "prim",
                   {"string", "name", "pos"},
                   {"enum float float3", "type", "float3"},
                   {"int","index","0"},
               }, /* outputs: */ {
                   "value",
               }, /* params: */ {
               }, /* category: */ {
                   "primitive",
               } });



///////////////////////////////////////////////////////////////////////////////
// 2022.07.11
///////////////////////////////////////////////////////////////////////////////

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Perlin Noise
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 参考来源：http://adrianb.io/2014/08/09/perlinnoise.html
// Hash lookup table as defined by Ken Perlin.  This is a randomly
// arranged array of all numbers from 0-255 inclusive.
// 第一步，我们需要先声明一个排列表（permutation table），或者称为 permutation[] 数组。
// 数组长度为256，分别随机、无重复地存放了 0-255 这些数值。为了避免缓存溢出，我们再重复填充一次数组的值，所以数组最终长度为 512。
// 这个 permutation[] 数组会在算法后续的哈希计算中使用到，用于确定一组输入最终挑选哪个梯度向量（从 12 个梯度向量中挑选）。
const int permutation[] = {
    151,160,137,91,90,15,
    131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
    190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
    88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
    77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
    102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
    135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
    5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
    223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
    129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
    251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
    49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
    138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,
    151,160,137,91,90,15,
    131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
    190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
    88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
    77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
    102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
    135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
    5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
    223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
    129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
    251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
    49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
    138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,
};

// 定义带缓入缓出的插值函数
float fade(float t) {
    // Fade function as defined by Ken Perlin.  This eases coordinate values
    // so that they will ease towards integral values.  This ends up smoothing
    // the final output.
    return t * t * t * (t * (t * 6 - 15) + 10);         // 6t^5 - 15t^4 + 10t^3
}

int inc(int num) {
    return num + 1;
}

// 梯度噪声与 Value 噪声的区别在于，Value噪声是对周围顶点上的 随机值 进行插值，而梯度噪声是对周围顶点上的 随机梯度 进行插值
// 对梯度进行插值，这里有一个问题需要解决，那就是对向量的插值，得到的结果肯定还是向量，而前面说过，噪声的输出结果应该是一个浮点数，
// 那么要怎么实现这二者的转换呢？
// 这里的做法是，将 当前像素点 到 周围顶点 的连线作为一个向量，与这个顶点的梯度进行点乘，就得到了对应的浮点数，
// 之后再对这个浮点数应用与 Value Noise 一样的插值算法，就能得到对应的噪声结果了。
// 也就是说，对周围顶点上的随机梯度，在当前位置点上的 投影 的长度 进行插值。
//
// grad()函数的作用在于计算随机选取的 梯度向量 以及 位置向量 的点积。
// 他们都是从以下 16 个向量里随机挑选一个(即哈希值模 16)作为梯度向量：
// (1,1,0),(-1,1,0),(1,-1,0),(-1,-1,0), 从坐标原点指向 4 个点，这 4 个点的坐标是 xy平面与单位立方体切割，割到的 4 个边的坐标
// (1,0,1),(-1,0,1),(1,0,-1),(-1,0,-1), 从坐标原点指向 4 个点，这 4 个点的坐标是 xz平面与单位立方体切割，割到的 4 个边的坐标
// (0,1,1),(0,-1,1),(0,1,-1),(0,-1,-1)  从坐标原点指向 4 个点，这 4 个点的坐标是 yz平面与单位立方体切割，割到的 4 个边的坐标
// (1,1,0),(-1,1,0),(0,-1,1),(0,-1,-1)  从坐标原点指向 4 个点，这 4 个点两两对角，构成一个正四面体
// 随机挑选结果其实取决于前一步所计算得出的哈希值（grad()函数的第一个参数）。
// 后面 3 个参数则代表由输入点指向顶点的距离向量（最终拿来与梯度向量进行点积）。
// dot( (x1,y1,z1), (x2,y2,z2) ) = x1*x2 + y1*y2 + z1*z2
// => dot( (1,1,0), (x,y,z) ) = x*1 + y*1 + z*0 = x + y
// 即 梯度向量 分别在各个 位置向量 上的投影
float grad(int hash, float x, float y, float z) {
    switch(hash & 0xF) {
    case 0x0: return  x + y;
    case 0x1: return -x + y;
    case 0x2: return  x - y;
    case 0x3: return -x - y;
    case 0x4: return  x + z;
    case 0x5: return -x + z;
    case 0x6: return  x - z;
    case 0x7: return -x - z;
    case 0x8: return  y + z;
    case 0x9: return -y + z;
    case 0xA: return  y - z;
    case 0xB: return -y - z;
    case 0xC: return  y + x;
    case 0xD: return -y + z;
    case 0xE: return  y - x;
    case 0xF: return -y - z;
    default: return 0;
    }
}

// perlin 噪声函数：
float PerlinNoise(float x, float y, float z) {

    // 首先，我们将输入坐标映射到 0.0 ~ 255.0 的范围内。
    x = fract(x / 256.f) * 256.f;
    y = fract(y / 256.f) * 256.f;
    z = fract(z / 256.f) * 256.f;

    // 接下来，我们将坐标映射到 [0,255] 范围内，它们代表了输入坐标落在了(256个单位正方体中的)哪个单位正方体中，
    // 这对应着 256 个随机值中的一个。这样我们以后访问数组时就不会遇到溢出错误。
    // 这也有一个不幸的副作用：Perlin 噪声总是每 256 个坐标重复一次。但这不是太大的问题，因为算法不仅能处理整数，还能处理小数。
    int xi = (int)x & 255;          // Calculate the "unit cube" that the point asked will be located in
    int yi = (int)y & 255;          // The left bound is ( |_x_|,|_y_|,|_z_| ) and the right bound is that
    int zi = (int)z & 255;          // plus 1.  Next we calculate the location (from 0.0 to 1.0) in that cube.

    // 最后，我们坐标映射到 0.0 ~ 1.0 范围内。
    float xf = x-(int)x;
    float yf = y-(int)y;
    float zf = z-(int)z;

    // 坐标映射到缓入缓出函数曲线上，这些 u / v / w 值稍后将用于插值。
    float u = fade(xf);
    float v = fade(yf);
    float w = fade(zf);

    // 柏林噪声 的 哈希函数 用于给每组输入计算返回一个唯一、确定值。哈希函数在维基百科的定义如下：
    //      哈希函数是一种从任何一种数据中创建小的数字 “指纹” 的方法，输入数据有任何细微的不同，都会令输出结果完全不一样
    // 这是 Perlin Noise 使用的哈希函数。它使用我们之前创建的表 permutation[]。
    // 输入坐标最终必定落入某个单位立方体中，哈希函数对这个单位立方体的 8 个顶点的索引号（xi,yi,zi）进行哈希计算
    // 由于哈希结果值是从 permutation[] 数组中得到的，所以哈希函数的返回值范围限定在 [0,255] 内。
    int aaa = permutation[permutation[permutation[    xi ]+    yi ]+    zi ];
    int aba = permutation[permutation[permutation[    xi ]+inc(yi)]+    zi ];
    int aab = permutation[permutation[permutation[    xi ]+    yi ]+inc(zi)];
    int abb = permutation[permutation[permutation[    xi ]+inc(yi)]+inc(zi)];
    int baa = permutation[permutation[permutation[inc(xi)]+    yi ]+    zi ];
    int bba = permutation[permutation[permutation[inc(xi)]+inc(yi)]+    zi ];
    int bab = permutation[permutation[permutation[inc(xi)]+    yi ]+inc(zi)];
    int bbb = permutation[permutation[permutation[inc(xi)]+inc(yi)]+inc(zi)];

    // 插值整合：经过前面的几步计算，我们得出了 8个顶点的影响值，并将它们进行平滑插值，得出了最终结果。
    float x1 = mix(    grad (aaa, xf  , yf  , zf),
                       grad (baa, xf-1, yf  , zf),
                       u);
    float x2 = mix(    grad (aba, xf  , yf-1, zf),
                       grad (bba, xf-1, yf-1, zf),
                       u);
    float y1 = mix(x1, x2, v);
    x1 = mix(    grad (aab, xf  , yf  , zf-1),
                 grad (bab, xf-1, yf  , zf-1),
                 u);
    x2 = mix(    grad (abb, xf  , yf-1, zf-1),
                 grad (bbb, xf-1, yf-1, zf-1),
                 u);
    float y2 = mix (x1, x2, v);

    return mix (y1, y2, w);
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Domain Warping
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 参考来源：https://iquilezles.org/articles/warp/
// 默认取 H=1.0, G=0.5 为佳
float fbm( vec3f pos, float H, float frequence, float amplitude, int numOctaves)
{
    float G = exp2(-H);
//    float G = 0.5;
    float t = 0.0;
    for( int i=0; i<numOctaves; i++ )
    {
        t += amplitude*PerlinNoise(frequence*pos[0],frequence*pos[1],frequence*pos[2]);
        frequence *= 2.0;
        amplitude *= G;
    }
    return t;
}

float domainWarpingV1( vec3f pos, float H, float frequence, float amplitude, int numOctaves)
{
    vec3f q = vec3f(fbm(pos + vec3f(0.0,0.0,0.0),H,frequence,amplitude,numOctaves),
                    fbm(pos + vec3f(1.7,2.8,9.2),H,frequence,amplitude,numOctaves),
                    fbm(pos + vec3f(5.2,8.3,1.3),H,frequence,amplitude,numOctaves));
    return fbm(pos + 4.0*q,H,frequence,amplitude,numOctaves);
}

struct PrimDomainWarpingV1Attr : INode {
    void apply() override {
        auto prim = has_input("prim") ?
                    get_input<PrimitiveObject>("prim") :
                    std::make_shared<PrimitiveObject>();

        auto H = get_input<NumericObject>("fbmH")->get<float>();
        auto frequence = get_input<NumericObject>("fbmFrequence")->get<float>();
        auto amplitude = get_input<NumericObject>("fbmAmplitude")->get<float>();
        auto numOctaves = get_input<NumericObject>("fbmNumOctaves")->get<int>();

        auto attrName = get_param<std::string>("attrName");
        auto attrType = get_param<std::string>("attrType");
        auto &pos = prim->verts;
        if (!prim->has_attr(attrName)) {
            if (attrType == "float3") prim->add_attr<vec3f>(attrName);
            else if (attrType == "float") prim->add_attr<float>(attrName);
        }

        prim->attr_visit(attrName, [&](auto &arr) {
#pragma omp parallel for
          for (int i = 0; i < arr.size(); i++) {
              if constexpr (is_decay_same_v<decltype(arr[i]), vec3f>) {
                  float x = domainWarpingV1(pos[i],H,frequence,amplitude,numOctaves);
                  float y = domainWarpingV1(pos[i],H,frequence,amplitude,numOctaves);
                  float z = domainWarpingV1(pos[i],H,frequence,amplitude,numOctaves);
                  arr[i] = vec3f(x,y,z);
              } else {
                  arr[i] = domainWarpingV1(pos[i],H,frequence,amplitude,numOctaves);
              }
          }
        });

        set_output("prim", get_input("prim"));
    }
};
ZENDEFNODE(PrimDomainWarpingV1Attr,
           { /* inputs: */ {
                   "prim",
                   {"float", "fbmH", "1.0"},
                   {"float", "fbmFrequence", "1.0"},
                   {"float", "fbmAmplitude", "1.0"},
                   {"int", "fbmNumOctaves", "4"},
               }, /* outputs: */ {
                   "prim",
               }, /* params: */ {
                   {"string", "attrName", "fnoise"},
                   {"enum float float3", "attrType", "float3"},
               }, /* category: */ {
                   "noise",
               }});

float domainWarpingV2( vec3f pos, float H, float frequence, float amplitude, int numOctaves)
{
    vec3f q = vec3f(fbm(pos + vec3f(0.0,0.0,0.0),H,frequence,amplitude,numOctaves),
                    fbm(pos + vec3f(1.7,2.8,9.2),H,frequence,amplitude,numOctaves),
                    fbm(pos + vec3f(5.2,8.3,1.3),H,frequence,amplitude,numOctaves));

    vec3f r = vec3f(fbm(pos + 4.0*q + vec3f(2.8,9.2,1.7),H,frequence,amplitude,numOctaves),
                    fbm(pos + 4.0*q + vec3f(9.2,1.7,2.8),H,frequence,amplitude,numOctaves),
                    fbm(pos + 4.0*q + vec3f(1.3,5.2,8.3),H,frequence,amplitude,numOctaves));

    return fbm(pos + 4.0*r,H,frequence,amplitude,numOctaves);
}

struct PrimDomainWarpingV2Attr : INode {
    void apply() override {
        auto prim = has_input("prim") ?
                    get_input<PrimitiveObject>("prim") :
                    std::make_shared<PrimitiveObject>();

        auto H = get_input<NumericObject>("fbmH")->get<float>();
        auto frequence = get_input<NumericObject>("fbmFrequence")->get<float>();
        auto amplitude = get_input<NumericObject>("fbmAmplitude")->get<float>();
        auto numOctaves = get_input<NumericObject>("fbmNumOctaves")->get<int>();

        auto attrName = get_param<std::string>("attrName");
        auto attrType = get_param<std::string>("attrType");
        auto &pos = prim->verts;
        if (!prim->has_attr(attrName)) {
            if (attrType == "float3") prim->add_attr<vec3f>(attrName);
            else if (attrType == "float") prim->add_attr<float>(attrName);
        }

        prim->attr_visit(attrName, [&](auto &arr) {
#pragma omp parallel for
          for (int i = 0; i < arr.size(); i++) {
              if constexpr (is_decay_same_v<decltype(arr[i]), vec3f>) {
                  float x = domainWarpingV2(pos[i],H,frequence,amplitude,numOctaves);
                  float y = domainWarpingV2(pos[i],H,frequence,amplitude,numOctaves);
                  float z = domainWarpingV2(pos[i],H,frequence,amplitude,numOctaves);
                  arr[i] = vec3f(x,y,z);
              } else {
                  arr[i] = domainWarpingV2(pos[i],H,frequence,amplitude,numOctaves);
              }
          }
        });

        set_output("prim", get_input("prim"));
    }
};
ZENDEFNODE(PrimDomainWarpingV2Attr,
           { /* inputs: */ {
                   "prim",
                   {"float", "fbmH", "1.0"},
                   {"float", "fbmFrequence", "1.0"},
                   {"float", "fbmAmplitude", "1.0"},
                   {"int", "fbmNumOctaves", "4"},
               }, /* outputs: */ {
                   "prim",
               }, /* params: */ {
                   {"string", "attrName", "fnoise"},
                   {"enum float float3", "attrType", "float3"},
               }, /* category: */ {
                   "noise",
               }});


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Simplex Noise
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// This method is a *lot* faster than using (int)Math.floor(x)
int fastfloor(double x) {
    return x>0 ? (int)x : (int)x-1;
}

// A lookup table to traverse the simplex around a given point in 4D.
// Details can be found where this table is used, in the 4D noise method.
const int simplex[][4] = {
    {0,1,2,3},{0,1,3,2},{0,0,0,0},{0,2,3,1},{0,0,0,0},{0,0,0,0},{0,0,0,0},{1,2,3,0},
    {0,2,1,3},{0,0,0,0},{0,3,1,2},{0,3,2,1},{0,0,0,0},{0,0,0,0},{0,0,0,0},{1,3,2,0},
    {0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},
    {1,2,0,3},{0,0,0,0},{1,3,0,2},{0,0,0,0},{0,0,0,0},{0,0,0,0},{2,3,0,1},{2,3,1,0},
    {1,0,2,3},{1,0,3,2},{0,0,0,0},{0,0,0,0},{0,0,0,0},{2,0,3,1},{0,0,0,0},{2,1,3,0},
    {0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},
    {2,0,1,3},{0,0,0,0},{0,0,0,0},{0,0,0,0},{3,0,1,2},{3,0,2,1},{0,0,0,0},{3,1,2,0},
    {2,1,0,3},{0,0,0,0},{0,0,0,0},{0,0,0,0},{3,1,0,2},{0,0,0,0},{3,2,0,1},{3,2,1,0}
};

float sGrad3(int hash, float x, float y, float z) {
    switch(hash & 0xF) {
    case 0x0: return  x + y;
    case 0x1: return -x + y;
    case 0x2: return  x - y;
    case 0x3: return -x - y;
    case 0x4: return  x + z;
    case 0x5: return -x + z;
    case 0x6: return  x - z;
    case 0x7: return -x - z;
    case 0x8: return  y + z;
    case 0x9: return -y + z;
    case 0xA: return  y - z;
    case 0xB: return -y - z;
    case 0xC: return  y + x;
    case 0xD: return -y + z;
    case 0xE: return  y - x;
    case 0xF: return -y - z;
    default: return 0;
    }
}

float sGrad4(int hash, float x, float y, float z, float w) {
    switch(hash & 0x1F) {
    case 0x00: return      y + z + w;
    case 0x01: return      y + z - w;
    case 0x02: return      y - z + w;
    case 0x03: return      y - z - w;
    case 0x04: return     -y + z + w;
    case 0x05: return     -y + z - w;
    case 0x06: return     -y - z + w;
    case 0x07: return     -y - z - w;

    case 0x08: return  x     + z + w;
    case 0x09: return  x     + z - w;
    case 0x0A: return  x     - z + w;
    case 0x0B: return  x     - z - w;
    case 0x0C: return -x     + z + w;
    case 0x0D: return -x     + z - w;
    case 0x0E: return -x     - z + w;
    case 0x0F: return -x     - z - w;

    case 0x10: return  x + y     + w;
    case 0x11: return  x + y     - w;
    case 0x12: return  x - y     + w;
    case 0x13: return  x - y     - w;
    case 0x14: return -x + y     + w;
    case 0x15: return -x + y     - w;
    case 0x16: return -x - y     + w;
    case 0x17: return -x - y     - w;

    case 0x18: return  x + y + z;
    case 0x19: return  x + y - z;
    case 0x1A: return  x - y + z;
    case 0x1B: return  x - y - z;
    case 0x1C: return -x + y + z;
    case 0x1D: return -x + y - z;
    case 0x1E: return -x - y + z;
    case 0x1F: return -x - y - z;
    default: return 0;
    }
}

// 3D Perlin simplex noise
// @param[in] x float coordinate
// @param[in] y float coordinate
// @param[in] z float coordinate
// @return Noise value in the range[-1; 1], value of 0 on all integer coordinates.
float SimplexNoise3(float x, float y, float z) {
    float n0, n1, n2, n3; // Noise contributions from the four corners

    // Skewing/Unskewing factors for 3D
    static const float F3 = 1.0f / 3.0f;
    static const float G3 = 1.0f / 6.0f;

    // Skew the input space to determine which simplex cell we're in
    float s = (x + y + z) * F3; // Very nice and simple skew factor for 3D
    int i = fastfloor(x + s);
    int j = fastfloor(y + s);
    int k = fastfloor(z + s);
    float t = (float)(i + j + k) * G3;
    float X0 = (float)i - t; // Unskew the cell origin back to (x,y,z) space
    float Y0 = (float)j - t;
    float Z0 = (float)k - t;
    float x0 = x - X0; // The x,y,z distances from the cell origin
    float y0 = y - Y0;
    float z0 = z - Z0;

    // For the 3D case, the simplex shape is a slightly irregular tetrahedron.
    // Determine which simplex we are in.
    int i1, j1, k1; // Offsets for second corner of simplex in (i,j,k) coords
    int i2, j2, k2; // Offsets for third corner of simplex in (i,j,k) coords
    if (x0 >= y0) {
        if (y0 >= z0) {
            i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 1; k2 = 0; // X Y Z order
        } else if (x0 >= z0) {
            i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 0; k2 = 1; // X Z Y order
        } else {
            i1 = 0; j1 = 0; k1 = 1; i2 = 1; j2 = 0; k2 = 1; // Z X Y order
        }
    } else { // x0<y0
        if (y0 < z0) {
            i1 = 0; j1 = 0; k1 = 1; i2 = 0; j2 = 1; k2 = 1; // Z Y X order
        } else if (x0 < z0) {
            i1 = 0; j1 = 1; k1 = 0; i2 = 0; j2 = 1; k2 = 1; // Y Z X order
        } else {
            i1 = 0; j1 = 1; k1 = 0; i2 = 1; j2 = 1; k2 = 0; // Y X Z order
        }
    }

    // A step of (1,0,0) in (i,j,k) means a step of (1-c,-c,-c) in (x,y,z),
    // a step of (0,1,0) in (i,j,k) means a step of (-c,1-c,-c) in (x,y,z), and
    // a step of (0,0,1) in (i,j,k) means a step of (-c,-c,1-c) in (x,y,z), where
    // c = 1/6.
    float x1 = x0 - (float)i1 + G3; // Offsets for second corner in (x,y,z) coords
    float y1 = y0 - (float)j1 + G3;
    float z1 = z0 - (float)k1 + G3;
    float x2 = x0 - (float)i2 + 2.0f * G3; // Offsets for third corner in (x,y,z) coords
    float y2 = y0 - (float)j2 + 2.0f * G3;
    float z2 = z0 - (float)k2 + 2.0f * G3;
    float x3 = x0 - 1.0f + 3.0f * G3; // Offsets for last corner in (x,y,z) coords
    float y3 = y0 - 1.0f + 3.0f * G3;
    float z3 = z0 - 1.0f + 3.0f * G3;

    // Wrap the integer indices at 256, to avoid indexing permutation[] out of bounds
    int ii = i & 0xff;
    int jj = j & 0xff;
    int kk = k & 0xff;

    // Work out the hashed gradient indices of the four simplex corners
    int gi0 = permutation[ii + permutation[jj + permutation[kk]]];
    int gi1 = permutation[ii + i1 + permutation[jj + j1 + permutation[kk + k1]]];
    int gi2 = permutation[ii + i2 + permutation[jj + j2 + permutation[kk + k2]]];
    int gi3 = permutation[ii + 1 + permutation[jj + 1 + permutation[kk + 1]]];

    // Calculate the contribution from the four corners
    float t0 = 0.6f - x0*x0 - y0*y0 - z0*z0;
//    float t0 = 0.5f - x0*x0 - y0*y0 - z0*z0;
    if (t0 < 0) {
        n0 = 0.0;
    } else {
        t0 *= t0;
        n0 = t0 * t0 * sGrad3(gi0, x0, y0, z0);
    }
    float t1 = 0.6f - x1*x1 - y1*y1 - z1*z1;
//    float t1 = 0.5f - x1*x1 - y1*y1 - z1*z1;
    if (t1 < 0) {
        n1 = 0.0;
    } else {
        t1 *= t1;
        n1 = t1 * t1 * sGrad3(gi1, x1, y1, z1);
    }
    float t2 = 0.6f - x2*x2 - y2*y2 - z2*z2;
//    float t2 = 0.5f - x2*x2 - y2*y2 - z2*z2;
    if (t2 < 0) {
        n2 = 0.0;
    } else {
        t2 *= t2;
        n2 = t2 * t2 * sGrad3(gi2, x2, y2, z2);
    }
    float t3 = 0.6f - x3*x3 - y3*y3 - z3*z3;
//    float t3 = 0.5f - x3*x3 - y3*y3 - z3*z3;
    if (t3 < 0) {
        n3 = 0.0;
    } else {
        t3 *= t3;
        n3 = t3 * t3 * sGrad3(gi3, x3, y3, z3);
    }
    // Add contributions from each corner to get the final noise value.
    // The result is scaled to stay just inside [-1,1]
    return 32.0f*(n0 + n1 + n2 + n3);
}

// 4D Perlin simplex noise
// @param[in] x float coordinate
// @param[in] y float coordinate
// @param[in] z float coordinate
// @param[in] w float coordinate
// @return Noise value in the range[-1; 1], value of 0 on all integer coordinates.
float SimplexNoise4(float x, float y, float z, float w) {

    float n0, n1, n2, n3, n4;   // Noise contributions from the five corners

    static const float F4 = 0.309016994f;   // F4 = (sqrt(5) - 1) / 4
    static const float G4 = 0.138196601f;   // G4 = (5 - sqrt(5)) / 20

    // Skew the (x,y,z,w) space to determine which cell of 24 simplices we're in
    float s = (x + y + z + w) * F4; // Factor for 4D skewing
    float xs = x + s;
    float ys = y + s;
    float zs = z + s;
    float ws = w + s;
    int i = fastfloor(xs);
    int j = fastfloor(ys);
    int k = fastfloor(zs);
    int l = fastfloor(ws);

    float t = (float)(i + j + k + l) * G4; // Factor for 4D unskewing
    float X0 = (float)i - t; // Unskew the cell origin back to (x,y,z,w) space
    float Y0 = (float)j - t;
    float Z0 = (float)k - t;
    float W0 = (float)l - t;

    float x0 = x - X0;  // The x,y,z,w distances from the cell origin
    float y0 = y - Y0;
    float z0 = z - Z0;
    float w0 = w - W0;

    // For the 4D case, the simplex is a 4D shape I won't even try to describe.
    // To find out which of the 24 possible simplices we're in, we need to
    // determine the magnitude ordering of x0, y0, z0 and w0.
    // The method below is a good way of finding the ordering of x,y,z,w and
    // then find the correct traversal order for the simplex we're in.
    // First, six pair-wise comparisons are performed between each possible pair
    // of the four coordinates, and the results are used to add up binary bits
    // for an integer index.
    int c1 = (x0 > y0) ? 32 : 0;
    int c2 = (x0 > z0) ? 16 : 0;
    int c3 = (y0 > z0) ? 8 : 0;
    int c4 = (x0 > w0) ? 4 : 0;
    int c5 = (y0 > w0) ? 2 : 0;
    int c6 = (z0 > w0) ? 1 : 0;
    int c = c1 + c2 + c3 + c4 + c5 + c6;

    int i1, j1, k1, l1; // The integer offsets for the second simplex corner
    int i2, j2, k2, l2; // The integer offsets for the third simplex corner
    int i3, j3, k3, l3; // The integer offsets for the fourth simplex corner

    // simplex[c] is a 4-vector with the numbers 0, 1, 2 and 3 in some order.
    // Many values of c will never occur, since e.g. x>y>z>w makes x<z, y<w and x<w
    // impossible. Only the 24 indices which have non-zero entries make any sense.
    // We use a thresholding to set the coordinates in turn from the largest magnitude.
    // The number 3 in the "simplex" array is at the position of the largest coordinate.
    i1 = simplex[c][0] >= 3 ? 1 : 0;
    j1 = simplex[c][1] >= 3 ? 1 : 0;
    k1 = simplex[c][2] >= 3 ? 1 : 0;
    l1 = simplex[c][3] >= 3 ? 1 : 0;
    // The number 2 in the "simplex" array is at the second largest coordinate.
    i2 = simplex[c][0] >= 2 ? 1 : 0;
    j2 = simplex[c][1] >= 2 ? 1 : 0;
    k2 = simplex[c][2] >= 2 ? 1 : 0;
    l2 = simplex[c][3] >= 2 ? 1 : 0;
    // The number 1 in the "simplex" array is at the second smallest coordinate.
    i3 = simplex[c][0] >= 1 ? 1 : 0;
    j3 = simplex[c][1] >= 1 ? 1 : 0;
    k3 = simplex[c][2] >= 1 ? 1 : 0;
    l3 = simplex[c][3] >= 1 ? 1 : 0;
    // The fifth corner has all coordinate offsets = 1, so no need to look that up.

    float x1 = x0 - (float)i1 + G4; // Offsets for second corner in (x,y,z,w) coords
    float y1 = y0 - (float)j1 + G4;
    float z1 = z0 - (float)k1 + G4;
    float w1 = w0 - (float)l1 + G4;
    float x2 = x0 - (float)i2 + 2.0f * G4; // Offsets for third corner in (x,y,z,w) coords
    float y2 = y0 - (float)j2 + 2.0f * G4;
    float z2 = z0 - (float)k2 + 2.0f * G4;
    float w2 = w0 - (float)l2 + 2.0f * G4;
    float x3 = x0 - (float)i3 + 3.0f * G4; // Offsets for fourth corner in (x,y,z,w) coords
    float y3 = y0 - (float)j3 + 3.0f * G4;
    float z3 = z0 - (float)k3 + 3.0f * G4;
    float w3 = w0 - (float)l3 + 3.0f * G4;
    float x4 = x0 - 1.0f + 4.0f * G4; // Offsets for last corner in (x,y,z,w) coords
    float y4 = y0 - 1.0f + 4.0f * G4;
    float z4 = z0 - 1.0f + 4.0f * G4;
    float w4 = w0 - 1.0f + 4.0f * G4;

    // Wrap the integer indices at 256, to avoid indexing permutation[] out of bounds
    int ii = i & 0xff;
    int jj = j & 0xff;
    int kk = k & 0xff;
    int ll = l & 0xff;

    // Calculate the contribution from the five corners
    float t0 = 0.6f - x0 * x0 - y0 * y0 - z0 * z0 - w0 * w0;
    if (t0 < 0.0) n0 = 0.0;
    else {
        t0 *= t0;
        n0 = t0 * t0 * sGrad4(permutation[ii + permutation[jj + permutation[kk + permutation[ll]]]], x0, y0, z0, w0);
    }

    float t1 = 0.6f - x1 * x1 - y1 * y1 - z1 * z1 - w1 * w1;
    if (t1 < 0.0) n1 = 0.0;
    else {
        t1 *= t1;
        n1 = t1 * t1 * sGrad4(permutation[ii + i1 + permutation[jj + j1 + permutation[kk + k1 + permutation[ll + l1]]]], x1, y1, z1, w1);
    }

    float t2 = 0.6f - x2 * x2 - y2 * y2 - z2 * z2 - w2 * w2;
    if (t2 < 0.0) n2 = 0.0;
    else {
        t2 *= t2;
        n2 = t2 * t2 * sGrad4(permutation[ii + i2 + permutation[jj + j2 + permutation[kk + k2 + permutation[ll + l2]]]], x2, y2, z2, w2);
    }

    float t3 = 0.6f - x3 * x3 - y3 * y3 - z3 * z3 - w3 * w3;
    if (t3 < 0.0) n3 = 0.0;
    else {
        t3 *= t3;
        n3 = t3 * t3 * sGrad4(permutation[ii + i3 + permutation[jj + j3 + permutation[kk + k3 + permutation[ll + l3]]]], x3, y3, z3, w3);
    }

    float t4 = 0.6f - x4 * x4 - y4 * y4 - z4 * z4 - w4 * w4;
    if (t4 < 0.0) n4 = 0.0;
    else {
        t4 *= t4;
        n4 = t4 * t4 * sGrad4(permutation[ii + 1 + permutation[jj + 1 + permutation[kk + 1 + permutation[ll + 1]]]], x4, y4, z4, w4);
    }

    // Sum up and scale the result to cover the range [-1,1]
    return 27.0f * (n0 + n1 + n2 + n3 + n4);
}

struct PrimSimplexNoiseAttr : INode {
    void apply() override {
        auto prim = has_input("prim") ?
                    get_input<PrimitiveObject>("prim") :
                    std::make_shared<PrimitiveObject>();

        auto attrName = get_param<std::string>("attrName");
        auto attrType = get_param<std::string>("attrType");
        auto &pos = prim->verts;
        if (!prim->has_attr(attrName)) {
            if (attrType == "float3") prim->add_attr<vec3f>(attrName);
            else if (attrType == "float") prim->add_attr<float>(attrName);
        }

        prim->attr_visit(attrName, [&](auto &arr) {
#pragma omp parallel for
          for (int i = 0; i < arr.size(); i++) {
              if constexpr (is_decay_same_v<decltype(arr[i]), vec3f>) {
                  vec3f p = pos[i];
                  float x = SimplexNoise3(p[0], p[1],p[2]);
                  float y = SimplexNoise3(p[1], p[2], p[0]);
                  float z = SimplexNoise3(p[2], p[0], p[1]);
                  arr[i] = vec3f(x,y,z);
              } else {
                  vec3f p = pos[i];
                  arr[i] = SimplexNoise3(p[0], p[1],p[2]);
              }
          }
        });

        set_output("prim", get_input("prim"));
    }
};
ZENDEFNODE(PrimSimplexNoiseAttr,
           { /* inputs: */ {
                   "prim",
               }, /* outputs: */ {
                   "prim",
               }, /* params: */ {
                   {"string", "attrName", "snoise"},
                   {"enum float float3", "attrType", "float3"},
               }, /* category: */ {
                   "noise",
               }});


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Worley Noise
// 参考来源：https://iquilezles.org/articles/smoothvoronoi/
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
glm::vec3 random3( glm::vec3 p ) {
    glm::vec3 val = sin(glm::vec3(dot(p,glm::vec3(127.1,311.7, 74.7)),
                                  dot(p,glm::vec3(269.5, 183.3, 246.1)),
                                  dot(p,glm::vec3(113.5, 271.9, 124.6))));
    val *= 43758.5453123;
    return fract(val);
}

float mydistance(glm::vec3 a, glm::vec3 b, int t) {
    if(t == 0) {
        return length(a - b);
    } else if (t == 1) {
        float xx = abs(a.x - b.x);
        float yy = abs(a.y - b.y);
        float zz = abs(a.z - b.z);
        return max(max(xx,yy),zz);
    } else {
        float xx = abs(a.x - b.x);
        float yy = abs(a.y - b.y);
        float zz = abs(a.z - b.z);
        return xx + yy + zz;
    }
}

float WorleyNoise3(float px, float py, float pz, int fType, int distType, float offsetX, float offsetY, float offsetZ) {
    // Tile the space
    glm::vec3 pos = glm::vec3(px,py,pz);   // 输入的位置坐标
    glm::vec3 offset = glm::vec3(offsetX, offsetY, offsetZ);    // 用于动画的偏移
    glm::vec3 i_pos = floor(pos);       // 输入的位置坐标所在的当前网格的索引
    glm::vec3 f_pos = fract(pos);       // 输入的位置坐标在当前网格中的相对位置

    float f1 = 9e9;
    float f2 = f1;

    // 以相对坐标遍历所有近邻网格（含当前网格）各个维度的范围都是（-1，0，1）
    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                // 每个近邻网格的原点的相对坐标
                glm::vec3 neighbor = glm::vec3(float(x),float(y),float(z));

                // 根据网格原点的全局坐标，每个临近网格都会产生一个随机位置，其范围是 (0.0, 0.0, 0.0) ~ (1.0, 1.0, 1.0)
                glm::vec3 point = random3(i_pos + neighbor);

                // 使用 offset 可以动画这个随机位置
                point = (float)0.5 + (float)0.5*sin(offset + (float)6.2831*point);

                // 每个临近网格的特征点的相对位置 = 临近网格的原点的相对坐标 + 随机位置
                glm::vec3 featurePoint = neighbor + point;

                // 特征点的相对位置 与 输入坐标的位置 之间的距离
                float dist = mydistance(featurePoint, f_pos, distType);

                if (dist < f1) {
                    f2 = f1; f1 = dist;     // 距离小于 “第一小距离”，则 “第二小距离” = “第一小距离”，“第一小距离” 存放最新的最小距离
                }else if(dist < f2) {
                    f2 = dist;              // 只是小于 “第二小距离” 而已，更新 “第二小距离”
                }                           // 比 “第二小距离” 还大，就什么都不用做了
            }
        }
    }

    if(fType == 0) {
        return f1;
    }
    else {
        return f2 - f1;
    }
}

struct PrimWorleyNoiseAttr : INode {
    void apply() override {
        auto prim = has_input("prim") ? get_input<PrimitiveObject>("prim") : std::make_shared<PrimitiveObject>();

        auto offset = vec3f(frand(), frand(), frand());
        if(has_input("seed")) offset = get_input<NumericObject>("seed")->get<vec3f>();

        int fType = 0;
        auto fTypeStr = get_input2<std::string>("fType");
//        if (fTypeStr == "F1"   ) fType = 0;
        if (fTypeStr == "F2-F1") fType = 1;

        int distType = 0;
        auto distTypeStr = get_input2<std::string>("distType");
//        if (distTypeStr == "Euclidean") distType = 0;
        if (distTypeStr == "Chebyshev") distType = 1;
        if (distTypeStr == "Manhattan") distType = 2;

        auto attrName = get_param<std::string>("attrName");
        auto attrType = get_param<std::string>("attrType");
        auto &pos = prim->verts;
        if (!prim->has_attr(attrName)) {
            if (attrType == "float3") prim->add_attr<vec3f>(attrName);
            else if (attrType == "float") prim->add_attr<float>(attrName);
        }

        prim->attr_visit(attrName, [&](auto &arr) {
#pragma omp parallel for
          for (int i = 0; i < arr.size(); i++) {
              if constexpr (is_decay_same_v<decltype(arr[i]), vec3f>) {
                  vec3f p = pos[i];
                  float x = WorleyNoise3(p[0],p[1],p[2],fType,distType,offset[0],offset[1],offset[2]);
                  float y = WorleyNoise3(p[1],p[2],p[0],fType,distType,offset[0],offset[1],offset[2]);
                  float z = WorleyNoise3(p[2],p[0],p[1],fType,distType,offset[0],offset[1],offset[2]);
                  arr[i] = vec3f(x,y,z);
              } else {
                  vec3f p = pos[i];
                  arr[i] = WorleyNoise3(p[0],p[1],p[2],fType,distType,offset[0],offset[1],offset[2]);
              }
          }
        });

        set_output("prim", get_input("prim"));
    }
};
ZENDEFNODE(PrimWorleyNoiseAttr,
           { /* inputs: */ {
                   "prim",
                   "seed",
                   {"enum Euclidean Chebyshev Manhattan", "distType", "Euclidean"},
                   {"enum F1 F2-F1", "fType", "F1"},
               }, /* outputs: */ {
                   "prim",
               }, /* params: */ {
                   {"string", "attrName", "wnoise"},
                   {"enum float float3", "attrType", "float"},
//                   {"enum Euclidean Chebyshev Manhattan", "distType", "Euclidean"},
//                   {"enum F1 F2-F1", "fType", "F1"},
               }, /* category: */ {
                   "noise",
               }});


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Voronoi
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void Voronoi3(const vec3f pos, const std::vector<vec3f>& points, float &voronoi, vec3f &minPoint) {
    float minDist = 9e9;
    for(auto const& point : points)
    {
        float dist = length(pos - point);
        if ( dist < minDist ) {
            minDist = dist;
            minPoint = point;
        }
    }
    voronoi = minDist;
}

struct PrimVoronoiAttr : INode {
    void apply() override {
        auto prim        = has_input("prim")        ? get_input<PrimitiveObject>("prim")        : std::make_shared<PrimitiveObject>();
        auto featurePrim = has_input("featurePrim") ? get_input<PrimitiveObject>("featurePrim") : std::make_shared<PrimitiveObject>();

        auto attrName = get_param<std::string>("attrName");
        if (!prim->has_attr(attrName)) {prim->add_attr<float>(attrName);}
        if (!prim->has_attr("minFeaturePointPos")) {prim->add_attr<vec3f>("minFeaturePointPos");}

        auto &attr_voro = prim->attr<float>(attrName);
        auto &attr_mFPP = prim->attr<vec3f>("minFeaturePointPos");

        auto &samplePoints = prim->verts;
        auto &featurePoints = featurePrim->verts;
#pragma omp parallel for
        for (int i = 0; i < prim->size(); i++) {
            Voronoi3(samplePoints[i],featurePrim->verts,attr_voro[i],attr_mFPP[i]);
        }

        set_output("prim", get_input("prim"));
    }
};
ZENDEFNODE(PrimVoronoiAttr,
           { /* inputs: */ {
                   "prim",
                   "featurePrim",
               }, /* outputs: */ {
                   "prim",
               }, /* params: */ {
                   {"string", "attrName", "voronoi"},
               }, /* category: */ {
                   "noise",
               }});



///////////////////////////////////////////////////////////////////////////////
// 2022.07.22
///////////////////////////////////////////////////////////////////////////////
struct BVHNearestPos : INode{
    void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto points = get_input<PrimitiveObject>("points");
        auto bvh_id  = points->attr<float>(get_input2<std::string>("bvhIdTag"));
        auto bvh_ws  = points->attr<vec3f>(get_input2<std::string>("bvhWeightTag"));
        auto &bvh_pos = points->add_attr<vec3f>( get_input2<std::string>("bvhPosTag"));

#pragma omp parallel for
        for (int i = 0; i < points->size(); i++) {
            vec3i vertsIdx = prim->tris[(int)bvh_id[i]];
            int v0 = vertsIdx[0], v1 = vertsIdx[1], v2 = vertsIdx[2];
            auto p0 = prim->attr<vec3f>("pos")[v0];
            auto p1 = prim->attr<vec3f>("pos")[v1];
            auto p2 = prim->attr<vec3f>("pos")[v2];
            bvh_pos[i] = bvh_ws[i][0] * p0 + bvh_ws[i][1] * p1 + bvh_ws[i][2]*p2;
        };

        set_output("oPoints", get_input("points"));
    }
};
ZENDEFNODE(BVHNearestPos,
           { /* inputs: */ {
                   "prim",
                   "points",
                   {"string", "bvhIdTag", "bvh_id"},
                   {"string", "bvhWeightTag", "bvh_ws"},
                   {"string", "bvhPosTag", "bvh_pos"},
               }, /* outputs: */ {
                   "oPoints"
               }, /* params: */ {
               }, /* category: */ {
                   "primitive"
               }});





}
}