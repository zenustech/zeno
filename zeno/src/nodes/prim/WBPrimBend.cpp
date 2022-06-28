#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/MatrixObject.h>
#include <zeno/utils/orthonormal.h>
#include <cmath>
#include <glm/gtx/quaternion.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
                float cb = std::cos(bend);
                float sb = std::sin(bend);
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

        set_output("prim", get_input("prim"));
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


// 指定段数和半径
struct CreateCircle : INode {
    void apply() override
    {
        auto seg = get_input<zeno::NumericObject>("segments")->get<int>();
        auto r = get_input<zeno::NumericObject>("r")->get<float>();
        auto prim = std::make_shared<zeno::PrimitiveObject>();

        for (int i = 0; i < seg; i++)
        {
            float rad = 2 * M_PI * i / seg;
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


// 指定起点朝向和结束朝向
struct QuatRotBetweenVectors : INode {
    void apply() override
    {
        auto start = get_input<zeno::NumericObject>("start")->get<zeno::vec3f>();
        auto dest = get_input<zeno::NumericObject>("dest")->get<zeno::vec3f>();

        glm::vec3 gl_start(start[0], start[1], start[2]);
        glm::vec3 gl_dest(dest[0], dest[1], dest[2]);

        zeno::vec4f rot;
        glm::quat quat = glm::rotation(gl_start, gl_dest);
        rot[0] = quat.x;
        rot[1] = quat.y;
        rot[2] = quat.z;
        rot[3] = quat.w;

        auto rotation = std::make_shared<zeno::NumericObject>();
        rotation->set<zeno::vec4f>(rot);
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
struct MatTranspose : zeno::INode{
    void apply() override
    {
        glm::mat mat = std::get<glm::mat4>(get_input<zeno::MatrixObject>("mat")->m);
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
        retprim->verts.resize(segments + 1);
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
            auto &cu = retprim->verts.attr<float>("curveU");
            cu[idx] = insertU;  // 设置新 prim 的属性
        }

        // 为新 prim 构建 lines
        retprim->lines.reserve(retprim->size());
        for (int i = 1; i < retprim->size(); i++) {
            retprim->lines.emplace_back(i - 1, i);
        }

        // 为新 prim 构建 points，因为 @rad 参数实际上可视化的是 points
        retprim->points.resize(retprim->size());
        for (int i = 0; i < retprim->size(); i++) {
            retprim->points[i] = i;
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
//            printf("wb-Debug ==> 01 line_index = %i\n", int(index));
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
//                        printf("wb-Debug ==> 02 line_index = %i\n", int(index));
                    prim->verts->erase(prim->verts.begin() + int(index) + 2, prim->verts.end());
                    cu.erase(cu.begin() + int(index) + 2, cu.end());
                }
            }
            else
            {
                if (prim->verts.begin() + int(index) <= prim->verts.end())
                {
//                        printf("wb-Debug ==> 02 line_index = %i\n", int(index));
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

        // 重建 points，因为 @rad 参数实际上可视化的是 points
        prim->points.resize(prim->size());
        for (int i = 0; i < prim->size(); i++) {
            prim->points[i] = i;
        }

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


// visualize vec3 attribute
struct VisVec3Attribute : INode {
    void apply() override
    {
        // 额外的标记
        int primOutPutType = 1; // 0 = ref, 1 = copy
        int originalPrimModified = 0;

        auto color = get_input<zeno::NumericObject>("color")->get<zeno::vec3f>();
        auto useNormalize = get_input<NumericObject>("normalize")->get<int>();
        auto lengthScale = get_input<NumericObject>("lengthScale")->get<float>();
        auto name = get_input2<std::string>("name");
        auto prim = get_input<PrimitiveObject>("prim");

        auto &it = prim->attr(name);
        std::vector<vec3f>& attr = std::get<std::vector<vec3f>>(it);

        // 构建新的可视化的 Prim
        auto primVis = std::make_shared<PrimitiveObject>();
        primVis->verts.resize(prim->size() * 2);
        primVis->lines.resize(prim->size());

        primVis->add_attr<vec3f>("clr");
        auto &itVisColor = primVis->attr("clr");
        std::vector<vec3f>& visColor = std::get<std::vector<vec3f>>(itVisColor);

#pragma omp parallel for
        for (int i = 0; i < primVis->verts.size(); i++)
        {
            // primVis
            int iRem = i % 2;   // 模 2 的余数，0 放起点位置，1 放终点位置
            int iPrim = (i - iRem)/2;   // prim 的索引
            if (iRem == 0)  // 设置起点
            {
                primVis->verts[i] = prim->verts[iPrim];
                visColor[i] = color;
            }
            else    // 设置终点
            {
                if (useNormalize)
                {
                    primVis->verts[i] = prim->verts[iPrim] + normalize(attr[iPrim]) * lengthScale;
                }
                else
                {
                    primVis->verts[i] = prim->verts[iPrim] + attr[iPrim] * lengthScale;
                }
                visColor[i] = color * 0.25;
                primVis->lines[i/2][0] = i-1;  // 线
                primVis->lines[i/2][1] = i;  // 线
            }
        }

        set_output("primVis", std::move(primVis));
    }
};
ZENDEFNODE(VisVec3Attribute,
           {  /* inputs: */ {
                   "prim",
                   {"string", "name", "NotUsePos"},
                   {"int", "normalize", "1"},
                   {"float", "lengthScale", "1.0"},
                   {"vec3f", "color", ""},
               }, /* outputs: */ {
                   "primVis",
               }, /* params: */ {
               }, /* category: */ {
                   "visualize",
               }});


// visualize vec3 field
struct TraceVec3FieldOneStep : INode {
    void apply() override
    {
        // 额外的标记
        int primOutPutType = 0; // 0 = ref, 1 = copy
        int originalPrimModified = 1;

        // 获取 name 属性值的引用
        auto prim = get_input<PrimitiveObject>("prim");
        auto name = get_input2<std::string>("name");
        auto &it = prim->attr(name);
        std::vector<vec3f>& attr = std::get<std::vector<vec3f>>(it);

        // OneStep 之前点和线的数量
        int oldVertsCount = prim->size();
        int oldlinesCount = prim->lines.size();

        // 为填入新的点和线准备空间
        auto vertsNum = get_input<NumericObject>("vertsNum")->get<int>();
        prim->verts.resize(oldVertsCount + vertsNum);
        prim->lines.resize(oldlinesCount + vertsNum);


#pragma omp parallel for
        for (int i = 0; i < vertsNum; i++)
        {
            // 本轮新点的位置 = 上一轮新点的位置 + 上一轮新点位置上采样到的场的属性值
            prim->verts[oldVertsCount + i] = (has_input("primB"))? get_input<PrimitiveObject>("primB")->verts[i] :
                                             prim->verts[oldVertsCount + i - vertsNum] + attr[oldVertsCount + i - vertsNum];

            // 新线的索引 = 上一轮新点的索引 : 本轮新点的索引
            prim->lines[oldlinesCount + i][0] = oldVertsCount + i - vertsNum;
            prim->lines[oldlinesCount + i][1] = oldVertsCount + i;
        }

        set_output("prim", std::move(prim));
    }
};
ZENDEFNODE(TraceVec3FieldOneStep,
           {  /* inputs: */ {
                   "prim",
                   "primB",
                   {"string", "name", "NotUsePos"},
                   {"int", "vertsNum", "0"},
               }, /* outputs: */ {
                   "prim",
               }, /* params: */ {
               }, /* category: */ {
                   "visualize",
               }});


// 创建属性并赋值或者为已有属性赋值
struct PrimCreatOrSetAttr : zeno::INode {
    void apply() override
    {
        int primOutPutType = 0; // 0 = ref, 1 = copy
        int originalPrimModified = 1;

        auto prim = get_input<PrimitiveObject>("prim");
        auto name = get_input2<std::string>("name");
        auto type = get_input2<std::string>("type");

        if (type == "float")
        {
            // 如果 fillValue 有值
            if (has_input("fillValue")) {
                auto fillvalue = get_input<NumericObject>("fillValue")->get<float>();

                if (!prim->has_attr(name)) {
                    // 如果这个属性不存在，创建这个属性并同时填入 fillValue
                    prim->add_attr<float>(name, fillvalue);
                } else {
                    // 如果这个属性已经存在，拿到存放这个属性的值的容器，
                    auto &val = prim->attr(name);
                    std::vector<float>& attr_arr = std::get<std::vector<float>>(val);

                    // 重新设置这个容器中的数据
#pragma omp parallel for
                    for (intptr_t i = 0; i < prim->verts.size(); i++) {
                        attr_arr[i] = fillvalue;
                    }
                }
            } else {
                // fillValue 中没有填值，此时，如果没有这个属性就创建这个属性
                if (!prim->has_attr(name)) {
                    prim->add_attr<float>(name);
                }
            }
        }
        else if (type == "float3")
        {
            if (has_input("fillValue")) {
                auto fillvalue = get_input<NumericObject>("fillValue")->get<vec3f>();
                if (!prim->has_attr(name)) {
                    // 如果这个属性不存在，创建这个属性并同时填入 fillValue
                    prim->add_attr<vec3f>(name, fillvalue);
                } else {
                    // 如果这个属性已经存在，拿到存放这个属性的值的容器，
                    auto &val = prim->attr(name);
                    std::vector<vec3f>& attr_arr = std::get<std::vector<vec3f>>(val);

                    // 重新设置这个容器中的数据
#pragma omp parallel for
                    for (intptr_t i = 0; i < prim->verts.size(); i++) {
                        attr_arr[i] = fillvalue;
                    }
                }
            } else {
                // fillValue 中没有填值，此时，如果没有这个属性就创建这个属性
                if (!prim->has_attr(name)) {
                    prim->add_attr<vec3f>(name);
                }
            }
        }
        else
        {
            throw Exception("Bad attribute type: " + type);
        }

//            set_output("prim", get_input("prim"));
        set_output("prim", std::move(prim));
    }
};
ZENDEFNODE(PrimCreatOrSetAttr,
           { /* inputs: */ {
                   "prim",
                   {"string", "name", "pos"},
                   {"enum float float3", "type", "float3"},
                   "fillValue",
               }, /* outputs: */ {
                   "prim",
               }, /* params: */ {
               }, /* category: */ {
                   "primitive",
               } });

}
}
