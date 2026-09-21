//============== Robot 节点：加载 MuJoCo ISF 静态 + 动态场景 ==============
//
// 原本的单体 RobotNode 被拆分为四个协同工作的节点：
//   1. RobotStaticNode      : 读取 scene_static.isf.json，解析 mesh/材质/纹理/geom，
//                             输出 prims（mesh 的 PrimitiveObject 列表）、
//                             matMap（mat_id → mtlidkey 映射）、materials（仅含 MaterialObject）。
//   2. RobotBindMaterialNode: 接收 RobotStaticNode 的 prims + matMap，将材质绑定
//                             到 prim 上（matid 属性，mtlid/matNum/Material_0），
//                             输出处理后的 prims。
//   3. RobotDynamicNode     : 读取 scene_dynamic.isf.json（manifest + frames/），
//                             输出全部动态帧信息（RobotDynamicData）。
//   4. RobotSceneNode       : 接收已绑定材质的 prims + 动态帧，
//                             通过 SceneAssembler 组装可渲染场景，输出 scene 列表
//                             （材质注册由外部节点单独处理）。
//
// 材质辅助函数（buildPBRFragShader、buildMaterialObject 等）位于
// RobotNodeMat.h（仅头文件）。静态场景解析管线（形状生成器、mesh/hfield 加载、
// 材质解析与 buildStaticScene）和动态数据构建核心（buildDynamicDataFromManifest）
// 实现为匿名命名空间自由函数，供磁盘路径与 ReadRobotSimulation 复用。
//
// 渲染管线说明：mesh 顶点保持在局部空间，世界变换以 Matrixes prim 形式提供，
// 渲染器可逐帧更新。

#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/MaterialObject.h>
#include <zeno/types/TextureObject.h>
#include <zeno/types/MatrixObject.h>
#include <functional>
#include <zeno/extra/SceneAssembler.h>
#include <zeno/extra/GlobalComm.h>
#include <zeno/extra/RobotNodeMat.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/utils/log.h>
#include <zeno/utils/scope_exit.h>
#include <tinygltf/json.hpp>
#include <glm/glm.hpp>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <map>
#include <vector>
#include <string>

#ifdef ZENO_WITH_PYTHON3
#include <Python.h>
#include <zeno/types/PythonContextObject.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace zeno {
namespace {

// ---- 辅助函数：将 MuJoCo xpos(3) + xmat(9, 行主序) 转换为 glm::mat4 ----
// RobotStaticNode（initial_global）和 RobotSceneNode（动态帧）共用。
static glm::mat4 makeMat4FromXposXmat(const std::vector<float>& xpos, const std::vector<float>& xmat) {
    glm::mat4 m(1.0f);
    // MuJoCo xmat 是行主序 3x3：[r00,r01,r02, r10,r11,r12, r20,r21,r22]
    // glm::mat4 是列主序：m[col][row]
    m[0] = glm::vec4(xmat[0], xmat[3], xmat[6], 0.0f);
    m[1] = glm::vec4(xmat[1], xmat[4], xmat[7], 0.0f);
    m[2] = glm::vec4(xmat[2], xmat[5], xmat[8], 0.0f);
    m[3] = glm::vec4(xpos[0], xpos[1], xpos[2], 1.0f);
    return m;
}

// ---- 结构体：材质映射（mat_id → mtlidkey），供 RobotBindMaterialNode 使用 ----
struct RobotMatMap : IObjectClone<RobotMatMap> {
    std::map<int, std::string> matIdToMtlid;
};

// ---- 结构体：动态节点输出（全部动态帧信息）----
struct RobotDynamicData : IObjectClone<RobotDynamicData> {
    std::string dynamicPath;             // manifest 路径（scene_dynamic.isf.json）
    RobotJson manifest;                  // 解析后的 manifest JSON
    int nFrames = -1;                    // 总帧数（无动态数据时为 -1）
    std::vector<RobotJson> frames;       // 全部解析后的帧 JSON（索引 = 帧顺序）
};

// ---- 动态数据公共构建核心（磁盘 RobotDynamicNode / 内存 ReadRobotSimulation 共用）----
// manifest 描述帧信息，frameLoader(i) 返回第 i 帧的 JSON：
//   磁盘路径 —— 根据 manifest.frames[].file（或旧命名约定）读取帧文件；
//   内存路径 —— 直接取内联的 frames[i]。
// 统一解析规则（覆盖磁盘旧版 manifest 与内存的单帧/内联形态）：
//   1) frames[] 非空        → 帧数 = 列表大小，逐帧 frameLoader
//   2) 顶层存在 transforms   → 单帧形态（manifest 即唯一帧），对应内存单帧
//   3) 存在 n_frames         → 帧数 = n_frames，逐帧 frameLoader（磁盘旧版）
//   4) 否则                 → 无帧数据（frames[] 存在记 0，否则 -1）
static std::shared_ptr<RobotDynamicData> buildDynamicDataFromManifest(
        const RobotJson& manifest,
        const std::string& dynamicPathLabel,
        const std::function<RobotJson(int)>& frameLoader) {
    auto out = std::make_shared<RobotDynamicData>();
    out->dynamicPath = dynamicPathLabel;
    out->manifest = manifest;

    int nFrames = -1;
    if (manifest.contains("frames") && manifest["frames"].is_array()
            && !manifest["frames"].empty()) {
        nFrames = static_cast<int>(manifest["frames"].size());
    } else if (manifest.contains("transforms") && !manifest["transforms"].is_null()) {
        out->frames.push_back(manifest);
        out->nFrames = 1;
        return out;
    } else if (manifest.contains("n_frames") && manifest["n_frames"].is_number_integer()) {
        nFrames = manifest["n_frames"].get<int>();
    }

    if (nFrames > 0) {
        out->nFrames = nFrames;
        out->frames.reserve(nFrames);
        for (int i = 0; i < nFrames; i++)
            out->frames.push_back(frameLoader(i));
    } else {
        out->nFrames = (manifest.contains("frames") && manifest["frames"].is_array()) ? 0 : -1;
    }
    return out;
}

// =====================================================================
// RobotStaticNode：解析静态 ISF 场景（scene_static.isf.json）。
//
// 读取 mesh、材质、纹理和 geom，构建局部空间 mesh primitive 及其世界变换
// （来自 initial_global），同时为每个 ISF 材质创建 MaterialObject。
// mat_id<0 的 geom（无材质引用的纯色几何）按其 material.rgba 创建纯色材质，
// 并以合成 mat_id（1000000+obj_id）注册到 matMap，绑定仍由 RobotBindMaterialNode 完成。
// sites 也作为可渲染图元处理（规范 §4.10，MuJoCo viewer 默认渲染 site，
// 如 car.xml 轮子两侧的绿色 '+' 装饰盒），以偏移 id（1000000+site_id）避免与 geom 冲突。
// flex（可变形体/软体，规范 §4.26）作为"拓扑静态 + 顶点逐帧更新"的网格处理，
// 以偏移 id（2000000+flex_id）入场景，顶点世界坐标、矩阵恒等，场景节点每帧替换顶点。
// 输出三个对象：
//   - prims      : 仅含 PrimitiveObject 的列表（材质不进入此处，走 materials 槽）
//   - matMap     : mat_id → mtlidkey 映射
//   - materials  : 仅含 MaterialObject 的列表
//
// 实际静态解析由下方匿名命名空间自由函数（buildStaticScene 及其管线）承担，
// 本节点与 ReadRobotSimulation 共用同一套实现，apply 仅负责从磁盘加载 JSON。
// =====================================================================

    // ---- 基础形状生成器（局部空间，变换之前）----

    static std::shared_ptr<PrimitiveObject> makePlane(float hx, float hy) {
        auto prim = std::make_shared<PrimitiveObject>();
        prim->verts.push_back(zeno::vec3f(-hx, -hy, 0));
        prim->verts.push_back(zeno::vec3f( hx, -hy, 0));
        prim->verts.push_back(zeno::vec3f( hx,  hy, 0));
        prim->verts.push_back(zeno::vec3f(-hx,  hy, 0));
        auto& nrm = prim->verts.add_attr<zeno::vec3f>("nrm");
        nrm.assign(4, zeno::vec3f(0, 0, 1));
        // UV = 对象空间顶点位置坐标（ISF §4.9.3 图元 UV 约定）：
        // 无显式 UV 的图元按对象空间位置生成纹理坐标，配合 frag 的
        // att_uv × texrepeat，棋盘格按世界单位平铺（每 1/texrepeat 单位一个周期）。
        auto& uv = prim->verts.add_attr<zeno::vec3f>("uv");
        uv.push_back(zeno::vec3f(-hx, -hy, 0));
        uv.push_back(zeno::vec3f( hx, -hy, 0));
        uv.push_back(zeno::vec3f( hx,  hy, 0));
        uv.push_back(zeno::vec3f(-hx,  hy, 0));
        prim->tris.push_back(zeno::vec3i(0, 2, 1));
        prim->tris.push_back(zeno::vec3i(0, 3, 2));
        return prim;
    }

    static std::shared_ptr<PrimitiveObject> makeBox(float hx, float hy, float hz) {
        auto prim = std::make_shared<PrimitiveObject>();
        // 24 个顶点：每面 4 个（复制以支持逐面法线和 UV）
        // 面顺序：-z, +z, -y, +y, -x, +x
        // MuJoCo box 普通 2D 纹理的 UV 布局（纹理上的 4x2 网格）：
        //   (0,0)-(0.25,0.5)    = -z 面
        //   (0.25,0)-(0.5,0.5)  = +z 面
        //   (0.5,0)-(0.75,0.5)  = -y 面
        //   (0.75,0)-(1,0.5)    = +y 面
        //   (0,0.5)-(0.25,1)    = -x 面
        //   (0.25,0.5)-(0.5,1)  = +x 面
        struct FaceDef { zeno::vec3f v[4]; zeno::vec3f n; float u0, v0, u1, v1; };
        FaceDef faces[6] = {
            // -z 面（底）
            {{{-hx,-hy,-hz},{ hx,-hy,-hz},{ hx, hy,-hz},{-hx, hy,-hz}},
             {0,0,-1}, 0.0f, 0.0f, 0.25f, 0.5f},
            // +z 面（顶）
            {{{-hx,-hy, hz},{ hx,-hy, hz},{ hx, hy, hz},{-hx, hy, hz}},
             {0,0, 1}, 0.25f, 0.0f, 0.5f, 0.5f},
            // -y 面（前）
            {{{-hx,-hy,-hz},{ hx,-hy,-hz},{ hx,-hy, hz},{-hx,-hy, hz}},
             {0,-1, 0}, 0.5f, 0.0f, 0.75f, 0.5f},
            // +y 面（后）
            {{{-hx, hy,-hz},{ hx, hy,-hz},{ hx, hy, hz},{-hx, hy, hz}},
             {0, 1, 0}, 0.75f, 0.0f, 1.0f, 0.5f},
            // -x 面（左）
            {{{-hx,-hy,-hz},{-hx, hy,-hz},{-hx, hy, hz},{-hx,-hy, hz}},
             {-1,0, 0}, 0.0f, 0.5f, 0.25f, 1.0f},
            // +x 面（右）
            {{{ hx,-hy,-hz},{ hx, hy,-hz},{ hx, hy, hz},{ hx,-hy, hz}},
             { 1,0, 0}, 0.25f, 0.5f, 0.5f, 1.0f},
        };
        auto& nrm = prim->verts.add_attr<zeno::vec3f>("nrm");
        auto& uv  = prim->verts.add_attr<zeno::vec3f>("uv");
        for (int f = 0; f < 6; f++) {
            int base = prim->verts.size();
            for (int i = 0; i < 4; i++) {
                prim->verts.push_back(faces[f].v[i]);
                nrm.push_back(faces[f].n);
            }
            // 普通 2D 纹理布局：v0=(u0,v0), v1=(u1,v0), v2=(u1,v1), v3=(u0,v1)
            uv.push_back(zeno::vec3f(faces[f].u0, faces[f].v0, 0));
            uv.push_back(zeno::vec3f(faces[f].u1, faces[f].v0, 0));
            uv.push_back(zeno::vec3f(faces[f].u1, faces[f].v1, 0));
            uv.push_back(zeno::vec3f(faces[f].u0, faces[f].v1, 0));
            // 两个三角形：(0,1,2) 和 (0,2,3)
            prim->tris.push_back(zeno::vec3i(base, base+1, base+2));
            prim->tris.push_back(zeno::vec3i(base, base+2, base+3));
        }
        return prim;
    }

    static std::shared_ptr<PrimitiveObject> makeSphere(float radius, int rows = 16, int cols = 32) {
        auto prim = std::make_shared<PrimitiveObject>();
        auto& nrm = prim->verts.add_attr<zeno::vec3f>("nrm");
        auto& uv  = prim->verts.add_attr<zeno::vec3f>("uv");
        // 顶部极点
        prim->verts.push_back(zeno::vec3f(0, 0, radius));
        nrm.push_back(zeno::vec3f(0, 0, 1));
        uv.push_back(zeno::vec3f(0, 1, 0));
        for (int r = 1; r < rows; r++) {
            float theta = M_PI * r / rows;
            for (int c = 0; c < cols; c++) {
                float phi = 2 * M_PI * c / cols;
                zeno::vec3f v(
                    radius * sin(theta) * cos(phi),
                    radius * sin(theta) * sin(phi),
                    radius * cos(theta));
                prim->verts.push_back(v);
                nrm.push_back(zeno::normalize(v));
                uv.push_back(zeno::vec3f(static_cast<float>(c) / cols, 1.0f - theta / M_PI, 0));
            }
        }
        int botPole = prim->verts.size();
        prim->verts.push_back(zeno::vec3f(0, 0, -radius));
        nrm.push_back(zeno::vec3f(0, 0, -1));
        uv.push_back(zeno::vec3f(0, 0, 0));
        for (int c = 0; c < cols; c++) {
            int n = (c + 1) % cols;
            prim->tris.push_back(zeno::vec3i(0, 1 + c, 1 + n));
        }
        for (int r = 0; r < rows - 2; r++) {
            for (int c = 0; c < cols; c++) {
                int n = (c + 1) % cols;
                int i0 = 1 + r * cols + c;
                int i1 = 1 + r * cols + n;
                int i2 = 1 + (r+1) * cols + c;
                int i3 = 1 + (r+1) * cols + n;
                prim->tris.push_back(zeno::vec3i(i0, i2, i1));
                prim->tris.push_back(zeno::vec3i(i1, i2, i3));
            }
        }
        for (int c = 0; c < cols; c++) {
            int n = (c + 1) % cols;
            int i0 = 1 + (rows-2) * cols + c;
            int i1 = 1 + (rows-2) * cols + n;
            prim->tris.push_back(zeno::vec3i(i0, botPole, i1));
        }
        return prim;
    }

    static std::shared_ptr<PrimitiveObject> makeEllipsoid(float a, float b, float c, int rows = 16, int cols = 32) {
        auto prim = makeSphere(1.0f, rows, cols);
        for (auto& v : prim->verts)
            v = zeno::vec3f(v[0]*a, v[1]*b, v[2]*c);
        return prim;
    }

    static std::shared_ptr<PrimitiveObject> makeCylinder(float radius, float halfLen, int segs = 32) {
        auto prim = std::make_shared<PrimitiveObject>();
        auto& nrm = prim->verts.add_attr<zeno::vec3f>("nrm");
        auto& uv  = prim->verts.add_attr<zeno::vec3f>("uv");
        // 程序化纹理坐标 = 对象空间位置（ISF §4.9.3 图元 UV 约定）：
        // u = 0.5·texrepeat[0]·x，v = −0.5·texrepeat[1]·y（x/y 为对象局部坐标，
        // 相位常数偏移自由）。本处只烤 0.5·pos−0.5，配合 frag 的 att_uv × texrepeat。
        auto posUV = [](float x, float y) {
            return zeno::vec3f(0.5f * x - 0.5f, -0.5f * y - 0.5f, 0.0f);
        };
        for (int i = 0; i < segs; i++) {
            float a = 2 * M_PI * i / segs;
            float x = radius*cos(a), y = radius*sin(a);
            prim->verts.push_back(zeno::vec3f(x, y, halfLen));
            nrm.push_back(zeno::vec3f(cos(a), sin(a), 0));
            uv.push_back(posUV(x, y));
        }
        for (int i = 0; i < segs; i++) {
            float a = 2 * M_PI * i / segs;
            float x = radius*cos(a), y = radius*sin(a);
            prim->verts.push_back(zeno::vec3f(x, y, -halfLen));
            nrm.push_back(zeno::vec3f(cos(a), sin(a), 0));
            uv.push_back(posUV(x, y));
        }
        int topC = 2 * segs;
        int botC = 2 * segs + 1;
        prim->verts.push_back(zeno::vec3f(0, 0,  halfLen));
        nrm.push_back(zeno::vec3f(0, 0, 1));
        uv.push_back(zeno::vec3f(-0.5f, -0.5f, 0));
        prim->verts.push_back(zeno::vec3f(0, 0, -halfLen));
        nrm.push_back(zeno::vec3f(0, 0, -1));
        uv.push_back(zeno::vec3f(-0.5f, -0.5f, 0));
        for (int i = 0; i < segs; i++) {
            int n = (i + 1) % segs;
            prim->tris.push_back(zeno::vec3i(topC, i, n));
            prim->tris.push_back(zeno::vec3i(segs+i, botC, segs+n));
            prim->tris.push_back(zeno::vec3i(n, i, segs+i));
            prim->tris.push_back(zeno::vec3i(n, segs+i, segs+n));
        }
        return prim;
    }

    static std::shared_ptr<PrimitiveObject> makeCapsule(float radius, float halfLen, int segs = 24, int hemiRings = 6) {
        auto prim = std::make_shared<PrimitiveObject>();
        // 为每个顶点提供解析法线与环形 UV（与其他图元一致）：法线保证光照正确，
        // UV 保证贴图可采样（缺 uv 时着色器会回落到 (0,0)，纹理贴图失真）。
        auto& nrm = prim->verts.add_attr<zeno::vec3f>("nrm");
        auto& uv  = prim->verts.add_attr<zeno::vec3f>("uv");
        float zBot = -halfLen - radius;
        float zTop =  halfLen + radius;
        float zSpan = zTop - zBot;
        auto vC = [&](float z) { return (z - zBot) / zSpan; };
        // 按环添加顶点；relZ = z - 对应球心 z（柱体段 relZ=0，径向法线；
        // 半球段 relZ 使其法线指向球心，消除绕序推导法线的脆弱性）。
        auto addRing = [&](float z, float rr, float relZ) -> int {
            int idx = prim->verts.size();
            for (int i = 0; i < segs; i++) {
                float a = 2 * M_PI * i / segs;
                float x = rr * cos(a), y = rr * sin(a);
                prim->verts.push_back(zeno::vec3f(x, y, z));
                nrm.push_back(zeno::normalize(zeno::vec3f(x, y, relZ)));
                uv.push_back(zeno::vec3f((float)i / segs, vC(z), 0));
            }
            return idx;
        };
        int botPole = prim->verts.size();
        prim->verts.push_back(zeno::vec3f(0, 0, zBot));
        nrm.push_back(zeno::vec3f(0, 0, -1));
        uv.push_back(zeno::vec3f(0, 0, 0));
        std::vector<int> rings;
        for (int r = 1; r <= hemiRings; r++) {
            float theta = (M_PI/2) * r / hemiRings;
            float z = -halfLen - radius * cos(theta);
            rings.push_back(addRing(z, radius * sin(theta), z + halfLen));
        }
        rings.push_back(addRing(halfLen, radius, 0.0f));
        for (int r = 1; r < hemiRings; r++) {
            float theta = (M_PI/2) * r / hemiRings;
            float z = halfLen + radius * sin(theta);
            rings.push_back(addRing(z, radius * cos(theta), z - halfLen));
        }
        int topPole = prim->verts.size();
        prim->verts.push_back(zeno::vec3f(0, 0, zTop));
        nrm.push_back(zeno::vec3f(0, 0, 1));
        uv.push_back(zeno::vec3f(0, 1, 0));
        for (int i = 0; i < segs; i++) {
            int n = (i + 1) % segs;
            prim->tris.push_back(zeno::vec3i(botPole, rings[0]+n, rings[0]+i));
        }
        for (int r = 0; r < (int)rings.size() - 1; r++) {
            for (int i = 0; i < segs; i++) {
                int n = (i + 1) % segs;
                prim->tris.push_back(zeno::vec3i(rings[r]+i, rings[r]+n, rings[r+1]+i));
                prim->tris.push_back(zeno::vec3i(rings[r+1]+i, rings[r]+n, rings[r+1]+n));
            }
        }
        int last = rings.back();
        for (int i = 0; i < segs; i++) {
            int n = (i + 1) % segs;
            prim->tris.push_back(zeno::vec3i(last+i, last+n, topPole));
        }
        return prim;
    }

    // ---- hfield：从高度场数据生成网格（规范 §4.7）----
    // 与 MuJoCo 渲染一致（render_context.c mjr_uploadHField / model_objects.cc）：
    // 顶点 (r,c)：x = size[0]*(2c/(ncol-1)-1)，y = size[1]*(2r/(nrow-1)-1)，
    // z = data[r*ncol+c]*size[2]；顶面网格 + 四条侧边（延伸到 z=-size[3]）。
    // 顶点在 hfield 局部空间（以原点为中心），geom 的 initial_global 负责世界放置。
    static std::shared_ptr<PrimitiveObject> makeHField(
            const RobotJson& hf, const std::vector<float>& data) {
        int nrow = hf.value("nrow", 0);
        int ncol = hf.value("ncol", 0);
        auto size = hf.value("size", std::vector<float>{0, 0, 0, 0});
        if (nrow < 2 || ncol < 2 || (int)data.size() < nrow * ncol)
            return nullptr;
        float sx = size.size() > 0 ? size[0] : 0.0f;
        float sy = size.size() > 1 ? size[1] : 0.0f;
        float sz = size.size() > 2 ? size[2] : 0.0f;
        float szbase = size.size() > 3 ? size[3] : 0.0f;

        auto prim = std::make_shared<PrimitiveObject>();
        auto& uv = prim->verts.add_attr<zeno::vec3f>("uv");
        auto idx = [&](int r, int c) { return r * ncol + c; };
        for (int r = 0; r < nrow; r++)
            for (int c = 0; c < ncol; c++) {
                float x = sx * (2.0f * c / (ncol - 1) - 1.0f);
                float y = sy * (2.0f * r / (nrow - 1) - 1.0f);
                float z = data[idx(r, c)] * sz;
                prim->verts.push_back(zeno::vec3f(x, y, z));
                uv.push_back(zeno::vec3f(c / (float)(ncol - 1), r / (float)(nrow - 1), 0));
            }
        // 顶面（每格两个三角形，绕序 (a,b,d),(b,c,d) → 法线 +z）
        for (int r = 0; r < nrow - 1; r++)
            for (int c = 0; c < ncol - 1; c++) {
                int a = idx(r, c), b = idx(r, c + 1);
                int d = idx(r + 1, c), cc = idx(r + 1, c + 1);
                prim->tris.push_back(zeno::vec3i(a, b, d));
                prim->tris.push_back(zeno::vec3i(b, cc, d));
            }
        // 四条侧边（顶面边界 → 底部 z=-szbase），绕序保证法线向外
        auto addSide = [&](const std::vector<std::pair<int, int>>& edges) {
            for (auto& [p0, p1] : edges) {
                auto t0 = prim->verts[p0], t1 = prim->verts[p1];
                int b = (int)prim->verts.size();
                prim->verts.push_back(zeno::vec3f(t0[0], t0[1], -szbase));
                prim->verts.push_back(zeno::vec3f(t1[0], t1[1], -szbase));
                uv.push_back(zeno::vec3f(0, 0, 0));
                uv.push_back(zeno::vec3f(0, 0, 0));
                prim->tris.push_back(zeno::vec3i(p0, p1, b));
                prim->tris.push_back(zeno::vec3i(p1, b + 1, b));
            }
        };
        std::vector<std::pair<int, int>> rightEdge, backEdge, leftEdge, frontEdge;
        for (int r = 0; r < nrow - 1; r++) rightEdge.push_back({idx(r, ncol - 1), idx(r + 1, ncol - 1)});
        for (int c = ncol - 1; c > 0; c--) backEdge.push_back({idx(nrow - 1, c), idx(nrow - 1, c - 1)});
        for (int r = nrow - 1; r > 0; r--) leftEdge.push_back({idx(r, 0), idx(r - 1, 0)});
        for (int c = 0; c < ncol - 1; c++) frontEdge.push_back({idx(0, c), idx(0, c + 1)});
        addSide(rightEdge); addSide(backEdge); addSide(leftEdge); addSide(frontEdge);

        // 平滑法线（面积加权累加后归一化）
        auto& nrm = prim->verts.add_attr<zeno::vec3f>("nrm");
        nrm.resize(prim->verts.size());
        std::vector<zeno::vec3f> acc(prim->verts.size(), zeno::vec3f(0, 0, 0));
        for (auto& t : prim->tris) {
            auto& v0 = prim->verts[t[0]]; auto& v1 = prim->verts[t[1]]; auto& v2 = prim->verts[t[2]];
            zeno::vec3f e1 = v1 - v0, e2 = v2 - v0;
            zeno::vec3f fn = zeno::cross(e1, e2);
            acc[t[0]] += fn; acc[t[1]] += fn; acc[t[2]] += fn;
        }
        for (size_t i = 0; i < nrm.size(); i++)
            nrm[i] = zeno::normalize(acc[i]);
        return prim;
    }

    // ---- 变换辅助函数（makeMesh 使用）----
    // 四元数 (w,x,y,z) -> 3x3 行主序旋转矩阵
    static void quatToMat3(const std::vector<float>& q, float R[9]) {
        // 防御缺维：不足 4 元时回退单位旋转，避免数组越界
        if (q.size() < 4) {
            R[0] = R[4] = R[8] = 1.0f;
            R[1] = R[2] = R[3] = R[5] = R[6] = R[7] = 0.0f;
            return;
        }
        float w = q[0], x = q[1], y = q[2], z = q[3];
        R[0] = 1 - 2*(y*y + z*z); R[1] = 2*(x*y - z*w);     R[2] = 2*(x*z + y*w);
        R[3] = 2*(x*y + z*w);     R[4] = 1 - 2*(x*x + z*z); R[5] = 2*(y*z - x*w);
        R[6] = 2*(x*z - y*w);     R[7] = 2*(y*z + x*w);     R[8] = 1 - 2*(x*x + y*y);
    }

    // 用 (R, pos) 变换 primitive 的所有顶点
    static void applyTransform(PrimitiveObject* prim, const float R[9], const std::vector<float>& pos) {
        float px = pos.size() > 0 ? pos[0] : 0.0f;
        float py = pos.size() > 1 ? pos[1] : 0.0f;
        float pz = pos.size() > 2 ? pos[2] : 0.0f;
        for (auto& v : prim->verts) {
            float x = R[0]*v[0] + R[1]*v[1] + R[2]*v[2] + px;
            float y = R[3]*v[0] + R[4]*v[1] + R[5]*v[2] + py;
            float z = R[6]*v[0] + R[7]*v[1] + R[8]*v[2] + pz;
            v = zeno::vec3f(x, y, z);
        }
    }

    // 将 pos(3) + quat(w,x,y,z) 转换为 glm::mat4
    static glm::mat4 makeMat4FromPosQuat(const std::vector<float>& pos, const std::vector<float>& quat) {
        float R[9];
        quatToMat3(quat, R);
        glm::mat4 m(1.0f);
        m[0] = glm::vec4(R[0], R[3], R[6], 0.0f);
        m[1] = glm::vec4(R[1], R[4], R[7], 0.0f);
        m[2] = glm::vec4(R[2], R[5], R[8], 0.0f);
        float px = pos.size() > 0 ? pos[0] : 0.0f;
        float py = pos.size() > 1 ? pos[1] : 0.0f;
        float pz = pos.size() > 2 ? pos[2] : 0.0f;
        m[3] = glm::vec4(px, py, pz, 1.0f);
        return m;
    }

    // ---- 从 OBJ 文本中解析 vn（顶点法线）----
    // zeno 通用 OBJ 解析器（parse_obj）不读取 vn 行，法线信息会丢失。
    // ISF 导出器写出的 OBJ 中 vn 与顶点一一对应（f a/a/a 约定），
    // 因此 vn 数量等于顶点数量时可直接按序对齐。
    static void parseObjVertexNormals(const char* data, size_t size, std::vector<zeno::vec3f>& out) {
        out.clear();
        const char* it = data;
        const char* eit = data + size;
        while (it < eit) {
            auto nit = std::find(it, eit, '\n');
            auto nnit = nit + 1;
            if (nit > it && nit[-1] == '\r')
                --nit;
            if ((nit - it) >= 3 && it[0] == 'v' && it[1] == 'n' && it[2] == ' ') {
                const char* p = it + 3;
                auto skipWs = [&]() {
                    while (p < nit && (*p == ' ' || *p == '\t')) ++p;
                };
                char* end;
                skipWs();
                float x = static_cast<float>(std::strtod(p, &end));
                p = end;
                skipWs();
                float y = static_cast<float>(std::strtod(p, &end));
                p = end;
                skipWs();
                float z = static_cast<float>(std::strtod(p, &end));
                out.emplace_back(x, y, z);
            }
            it = nnit;
        }
    }

    // ---- 翻转 UV 的 v 坐标（v'=1-v）----
    // ISF 约定 mesh 纹理坐标 v=0 对应图像顶部（与源引擎输出/PNG 行序一致），
    // 而 zeno 用 stbi_set_flip_vertically_on_load(true) 加载纹理（v=0 对应图像底部）。
    // 渲染端负责把 ISF 的 UV 方向适配到自己的采样约定，此为通用适配。
    // cube 图集 UV（shader 内按对象空间方向采样）与 primitive 程序生成 UV 不走此路径。
    static void flipUVVertical(PrimitiveObject* prim) {
        if (prim->tris.has_attr("uv0") && prim->tris.has_attr("uv1") && prim->tris.has_attr("uv2")) {
            for (auto& v : prim->tris.attr<zeno::vec3f>("uv0")) v[1] = 1.0f - v[1];
            for (auto& v : prim->tris.attr<zeno::vec3f>("uv1")) v[1] = 1.0f - v[1];
            for (auto& v : prim->tris.attr<zeno::vec3f>("uv2")) v[1] = 1.0f - v[1];
        }
        if (prim->verts.has_attr("uv")) {
            for (auto& v : prim->verts.attr<zeno::vec3f>("uv")) v[1] = 1.0f - v[1];
        }
    }

    // ---- 通用字段加载：JSON 数组 或 *_b64（float32/int32 小端展平 + base64）二选一 ----
    // 静态 flex/skin 的 xpos/vertices/faces 与 mesh 内联同约定（ISF 4.6.5）：
    // 大顶点数（>=200000）时导出端输出 *_b64，此处两种形态都支持。
    // 返回 false 表示字段缺失/null/长度非法（调用方按"无该字段"处理）。
    static bool loadFieldVerts(const RobotJson& j, const char* key, PrimitiveObject* prim) {
        if (j.contains(key) && j[key].is_array()) {
            for (auto& v : j[key])
                prim->verts.push_back(zeno::vec3f(
                    v[0].get<float>(), v[1].get<float>(), v[2].get<float>()));
            return true;
        }
        std::string b64key = std::string(key) + "_b64";
        if (j.contains(b64key) && j[b64key].is_string()) {
            auto bytes = base64Decode(j[b64key].get<std::string>());
            if (bytes.size() % (3 * sizeof(float)) != 0)
                return false;
            size_t nv = bytes.size() / (3 * sizeof(float));
            prim->verts.resize(nv);
            std::memcpy(prim->verts.data(), bytes.data(), bytes.size());
            return true;
        }
        return false;
    }

    static bool loadFieldTris(const RobotJson& j, const char* key, PrimitiveObject* prim) {
        if (j.contains(key) && j[key].is_array()) {
            for (auto& f : j[key])
                prim->tris.push_back(zeno::vec3i(
                    f[0].get<int>(), f[1].get<int>(), f[2].get<int>()));
            return true;
        }
        std::string b64key = std::string(key) + "_b64";
        if (j.contains(b64key) && j[b64key].is_string()) {
            auto bytes = base64Decode(j[b64key].get<std::string>());
            if (bytes.size() % (3 * sizeof(int32_t)) != 0)
                return false;
            size_t nf = bytes.size() / (3 * sizeof(int32_t));
            prim->tris.resize(nf);
            std::memcpy(prim->tris.data(), bytes.data(), bytes.size());
            return true;
        }
        return false;
    }

    // ---- 构建 mesh primitive（优先 OBJ，回退到内联 JSON）----
    // ISF 1.0 将未缩放的资产顶点存储在 OBJ/vertices 中。所需的处理流水线为：
    //   scale -> offset (pos/quat) -> geom 世界变换。
    // 详见 ISF_FORMAT_SPEC.md §4.6.1。
    static std::shared_ptr<PrimitiveObject> makeMesh(const RobotJson& meshJson, const std::filesystem::path& baseDir) {
        auto prim = std::make_shared<PrimitiveObject>();
        // baseDir 为空（ReadRobotSimulation 从 python 上下文读内联数据）
        // 时不再尝试从磁盘加载 OBJ，直接走内联 vertices/faces 路径。
        if (meshJson.contains("obj_path") && !baseDir.empty()) {
            auto objPath = baseDir / std::filesystem::u8path(meshJson["obj_path"].get<std::string>());
            std::ifstream file(objPath, std::ios::binary);
            if (file.is_open()) {
                std::vector<char> buf((std::istreambuf_iterator<char>(file)),
                                      std::istreambuf_iterator<char>());
                auto* raw = primParsedFrom(buf.data(), buf.size());
                if (raw) {
                    prim.reset(raw);
                    primTriangulate(prim.get());
                    // 补充 OBJ 顶点法线（通用解析器不读取 vn）
                    std::vector<zeno::vec3f> objNrm;
                    parseObjVertexNormals(buf.data(), buf.size(), objNrm);
                    if (objNrm.size() == prim->verts.size()) {
                        auto& nrm = prim->verts.add_attr<zeno::vec3f>("nrm");
                        nrm.assign(objNrm.begin(), objNrm.end());
                    }
                }
            }
        }
        if (prim->verts.size() == 0) {
            // 回退：内联几何数据（JSON 数组 或 *_b64 大编码，二选一）。
            // *_b64 为 float32/int32 小端展平 + base64（导出器对 200000~5000000
            // 顶点的 mesh 使用，体积约减半）；>=5000000 顶点不内联（依赖 obj_path）。
            bool hasInlineB64 = meshJson.contains("vertices_b64") && meshJson["vertices_b64"].is_string()
                             && meshJson.contains("faces_b64") && meshJson["faces_b64"].is_string();
            bool hasInlineJson = meshJson.contains("vertices") && !meshJson["vertices"].empty()
                              && meshJson.contains("faces") && !meshJson["faces"].empty();
            if (hasInlineB64) {
                auto vbytes = base64Decode(meshJson["vertices_b64"].get<std::string>());
                auto fbytes = base64Decode(meshJson["faces_b64"].get<std::string>());
                if (vbytes.size() % (3 * sizeof(float)) == 0 && fbytes.size() % (3 * sizeof(int32_t)) == 0) {
                    size_t nv = vbytes.size() / (3 * sizeof(float));
                    size_t nf = fbytes.size() / (3 * sizeof(int32_t));
                    prim->verts.resize(nv);
                    std::memcpy(prim->verts.data(), vbytes.data(), vbytes.size());
                    prim->tris.resize(nf);
                    std::memcpy(prim->tris.data(), fbytes.data(), fbytes.size());
                    // 内联 normals_b64：float32 展平，与顶点一一对应
                    if (meshJson.contains("normals_b64") && meshJson["normals_b64"].is_string()) {
                        auto nbytes = base64Decode(meshJson["normals_b64"].get<std::string>());
                        if (nbytes.size() == nv * 3 * sizeof(float)) {
                            auto& nrm = prim->verts.add_attr<zeno::vec3f>("nrm");
                            nrm.resize(nv);
                            std::memcpy(nrm.data(), nbytes.data(), nbytes.size());
                        }
                    }
                    // 内联 texcoords_b64：float32 展平，每顶点 2 分量 → 补 0 成 vec3f
                    if (meshJson.contains("texcoords_b64") && meshJson["texcoords_b64"].is_string()) {
                        auto tbytes = base64Decode(meshJson["texcoords_b64"].get<std::string>());
                        if (tbytes.size() == nv * 2 * sizeof(float)) {
                            auto& uv = prim->verts.add_attr<zeno::vec3f>("uv");
                            uv.resize(nv);
                            const float* src = reinterpret_cast<const float*>(tbytes.data());
                            for (size_t i = 0; i < nv; i++)
                                uv[i] = zeno::vec3f(src[2*i], src[2*i+1], 0.0f);
                        }
                    }
                } else {
                    log_warn("RobotNode: mesh {} 的 *_b64 数据长度非法，已忽略",
                             meshJson.value("name", std::string("?")));
                }
            }
            if (prim->verts.size() == 0 && hasInlineJson) {
                for (auto& v : meshJson["vertices"]) {
                    prim->verts.push_back(zeno::vec3f(v[0].get<float>(), v[1].get<float>(), v[2].get<float>()));
                }
                for (auto& f : meshJson["faces"]) {
                    prim->tris.push_back(zeno::vec3i(f[0], f[1], f[2]));
                }
                // 内联 normals 数组存在时按顶点对齐重建 nrm 属性
                if (meshJson.contains("normals") && !meshJson["normals"].empty()
                        && meshJson["normals"].size() == prim->verts.size()) {
                    auto& nrm = prim->verts.add_attr<zeno::vec3f>("nrm");
                    nrm.resize(prim->verts.size());
                    for (size_t i = 0; i < prim->verts.size(); i++) {
                        nrm[i] = zeno::vec3f(
                            meshJson["normals"][i][0].get<float>(),
                            meshJson["normals"][i][1].get<float>(),
                            meshJson["normals"][i][2].get<float>());
                    }
                }
                // 内联 UV：texcoords 与顶点一一对应时按顶点设 uv
                // （per-vertex；per-corner 的 mesh 应优先用 obj_path 加载）。
                if (meshJson.contains("texcoords") && !meshJson["texcoords"].empty()
                        && meshJson["texcoords"].size() == prim->verts.size()) {
                    auto& uv = prim->verts.add_attr<zeno::vec3f>("uv");
                    uv.resize(prim->verts.size());
                    for (size_t i = 0; i < prim->verts.size(); i++) {
                        uv[i] = zeno::vec3f(
                            meshJson["texcoords"][i][0].get<float>(),
                            meshJson["texcoords"][i][1].get<float>(), 0.0f);
                    }
                }
            }
        }

        // 解析 mesh 资产变换：scale（顶点缩放）+ offset（旋转 R + 平移 pos）
        std::vector<float> scale = {1.0f, 1.0f, 1.0f};
        if (meshJson.contains("scale")) {
            auto s = meshJson["scale"].get<std::vector<float>>();
            if (s.size() >= 3) scale = s;
            else log_warn("RobotNode: mesh scale must contain 3 values");
        }
        float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        std::vector<float> pos = {0.0f, 0.0f, 0.0f};
        if (meshJson.contains("offset") && prim->verts.size() > 0) {
            auto& off = meshJson["offset"];
            if (off.contains("pos") && off.contains("quat")) {
                pos = off["pos"].get<std::vector<float>>();
                auto quat = off["quat"].get<std::vector<float>>();
                quatToMat3(quat, R);
            } else {
                // 缺失时回退为不偏移，避免 [] 访问抛异常中断整场景
                log_warn("RobotNode: mesh {} 的 offset 缺少 pos/quat，已忽略该偏移",
                         meshJson.value("name", std::string("?")));
            }
        }

        if (prim->verts.size() > 0) {
            // 1. scale（ISF 顶点为未缩放资产顶点，见 ISF_FORMAT_SPEC.md §4.6.1）
            for (auto& v : prim->verts) {
                v[0] *= scale[0];
                v[1] *= scale[1];
                v[2] *= scale[2];
            }
            // 2. offset（旋转 + 平移）
            applyTransform(prim.get(), R, pos);
            // 3. 法线按逆转置变换：n' = normalize(S^-1 · R^T · n)
            if (prim->verts.has_attr("nrm")) {
                float sInv[3] = {
                    std::fabs(scale[0]) > 1e-12f ? 1.0f / scale[0] : 1.0f,
                    std::fabs(scale[1]) > 1e-12f ? 1.0f / scale[1] : 1.0f,
                    std::fabs(scale[2]) > 1e-12f ? 1.0f / scale[2] : 1.0f,
                };
                auto& nrm = prim->verts.attr<zeno::vec3f>("nrm");
                for (auto& n : nrm) {
                    float x = R[0]*n[0] + R[3]*n[1] + R[6]*n[2];
                    float y = R[1]*n[0] + R[4]*n[1] + R[7]*n[2];
                    float z = R[2]*n[0] + R[5]*n[1] + R[8]*n[2];
                    x *= sInv[0]; y *= sInv[1]; z *= sInv[2];
                    float len = std::sqrt(x*x + y*y + z*z);
                    if (len > 1e-8f)
                        n = zeno::vec3f(x/len, y/len, z/len);
                }
            }
            // 4. 翻转 UV 的 v：ISF v=0 在图像顶部、zeno 在底部（详见 flipUVVertical）
            flipUVVertical(prim.get());
        }
        return prim;
    }

    // ---- 按 ISF type/type_id 生成基础图元（geom/site 共用）----
    // 返回 nullptr 表示类型不支持（调用方跳过并提示）。
    // 尺寸均为半尺寸（half-extents），见 ISF_FORMAT_SPEC.md §4.9.3。
    static std::shared_ptr<PrimitiveObject> makePrimitiveByType(
            const std::string& type, int typeId, const std::vector<float>& sizes) {
        if (type == "plane" || typeId == 0) {
            // MuJoCo plane size[0]=0 表示无限大；使用默认大尺寸
            float hx = (sizes.size() > 0 && sizes[0] > 0) ? sizes[0] : 50.0f;
            float hy = (sizes.size() > 1 && sizes[1] > 0) ? sizes[1] : 50.0f;
            return makePlane(hx, hy);
        } else if (type == "sphere" || typeId == 2) {
            // MuJoCo 编译默认 geom size = [0.05, 0.05, 0.05]
            return makeSphere(sizes.size() > 0 ? sizes[0] : 0.05f);
        } else if (type == "capsule" || typeId == 3) {
            return makeCapsule(sizes.size() > 0 ? sizes[0] : 0.05f,
                               sizes.size() > 1 ? sizes[1] : 0.0f);
        } else if (type == "ellipsoid" || typeId == 4) {
            float a = sizes.size() > 0 ? sizes[0] : 0.05f;
            float b = sizes.size() > 1 ? sizes[1] : a;
            float c = sizes.size() > 2 ? sizes[2] : a;
            return makeEllipsoid(a, b, c);
        } else if (type == "cylinder" || typeId == 5) {
            return makeCylinder(sizes.size() > 0 ? sizes[0] : 0.05f,
                                sizes.size() > 1 ? sizes[1] : 0.0f);
        } else if (type == "box" || typeId == 6) {
            float a = sizes.size() > 0 ? sizes[0] : 0.05f;
            float b = sizes.size() > 1 ? sizes[1] : a;
            float c = sizes.size() > 2 ? sizes[2] : a;
            return makeBox(a, b, c);
        }
        return nullptr;
    }

    // ---- 解析 geom/site 的 material，返回应存入 userData 的 mat_id ----
    // mat_id>=0：直接引用 materials 表，绑定由 RobotBindMaterialNode 完成。
    // mat_id<0（无材质引用的纯色几何，规范 §4.9.8）：用 material.rgba 创建
    // 纯色材质并注册到 matMap/materials（key 带 sceneKey 前缀保证跨场景唯一；
    // 合成 id 避免与真实材质 id 冲突），绑定仍由 RobotBindMaterialNode 按
    // matMap 解析。
    static int resolveMaterial(
            const RobotJson& materialJson, int objId,
            const std::shared_ptr<ListObject>& matList,
            const std::shared_ptr<RobotMatMap>& matMapOut,
            const std::string& sceneKey) {
        int matId = materialJson.value("mat_id", -1);
        if (matId >= 0)
            return matId;
        int solidId = 1000000 + objId;
        // geom id 每个场景从 0 起算，跨场景必重复；key 加场景前缀后唯一
        std::string solidKey = "robot_mat_solid_" + sceneKey + "_" + std::to_string(objId);
        vec4f rgba(0.5f, 0.5f, 0.5f, 1.0f);  // 防御：缺 rgba 时用默认灰
        if (materialJson.contains("rgba")) {
            auto g = materialJson["rgba"].get<std::vector<float>>();
            if (g.size() >= 4)
                rgba = vec4f(g[0], g[1], g[2], g[3]);
        }
        auto mtl = buildSolidColorMaterial(rgba, solidKey);
        matList->arr.push_back(mtl);
        matMapOut->matIdToMtlid[solidId] = solidKey;
        return solidId;
    }

    // ---- 构建静态场景（磁盘 RobotStaticNode / 内存 ReadRobotSimulation 共用）----
    // 输入 ISF 静态 JSON + 资源基础目录 baseDir，结果写入 primList、
    // matList、matMapOut。baseDir 为空表示资源全部内联
    // （mesh 走 vertices/faces，纹理走 data_base64，对应 ReadRobotSimulation）。
    static void buildStaticScene(
            const RobotJson& scene,
            const std::filesystem::path& baseDir,
            const std::shared_ptr<ListObject>& primList,
            const std::shared_ptr<ListObject>& matList,
            const std::shared_ptr<RobotMatMap>& matMapOut) {

        // ---- 场景唯一前缀 ----
        // ISF 根 name 是场景唯一标识（导出器用模型文件名生成，如
        // zenoBullet_duck_vhacd）。mtlidkey/solidKey 以其为前缀：
        // 渲染器按 key 去重/缓存材质，不同场景的同名材质（如 duck 的
        // "yellow"）若无前缀会互相覆盖，导致纹理/颜色错乱甚至崩溃。
        std::string sceneKey = scene.value("name", std::string(""));
        if (sceneKey.empty())
            sceneKey = "default";

        // ---- 加载 mesh ----
        std::map<int, RobotJson> meshMap;
        if (scene.contains("meshes")) {
            for (auto& m : scene["meshes"])
                meshMap[m["id"].get<int>()] = m;
        }

        // ---- 加载 hfield（高度场，规范 §4.7）----
        std::map<int, RobotJson> hfieldMap;
        if (scene.contains("hfields")) {
            for (auto& h : scene["hfields"])
                hfieldMap[h["id"].get<int>()] = h;
        }

        // ---- 加载材质和纹理 ----
        std::map<int, RobotJson> matMap;
        if (scene.contains("materials")) {
            for (auto& m : scene["materials"])
                matMap[m["id"].get<int>()] = m;
        }
        std::map<int, RobotJson> texMap;
        if (scene.contains("textures")) {
            for (auto& t : scene["textures"])
                texMap[t["id"].get<int>()] = t;
        }

        // ---- 无 geom 不早退：场景可能是纯 flex/skin/site（纯软体等）
        // 构成，材质与这些图元仍需构建。geom 循环内部用空数组守护缺失情况。

        // ---- 为每个 ISF 材质创建 MaterialObject ----
        // mtlidkey 以材质 id 为主体生成：MuJoCo 允许多个材质同名，
        // 仅用 name 会导致 mtlidkey 冲突，渲染器按 key 去重时丢失材质。
        // 材质只进 matList（磁盘路径与 ReadRobotSimulation 共用此解析，
        // prims 输出不含 MaterialObject，材质有专用 materials 槽）。
        for (auto& [mid, m] : matMap) {
            std::string mtlName = m.value("name", std::string(""));
            // sceneKey 前缀：跨场景材质 key 唯一。MuJoCo 允许多个材质同名，
            // 仅用 name 会冲突；渲染器按 key 去重时丢失材质。加场景前缀后
            // 不同场景同名材质天然区分，互不覆盖。
            std::string mtlidkey = (mtlName.empty()
                ? ("robot_mat_" + sceneKey + "_" + std::to_string(mid))
                : ("robot_mat_" + sceneKey + "_" + std::to_string(mid) + "_" + mtlName));
            auto mtl = buildMaterialObject(m, texMap, baseDir, mtlidkey);
            matList->arr.push_back(mtl);
            matMapOut->matIdToMtlid[mid] = mtlidkey;
        }

        // ---- 处理 geom：创建局部空间 mesh + 计算世界变换 ----
        // ISF 1.0 每个 geom 使用两个正交维度：
        //   source: {kind, type, data_id, size, collision_representation}
        //   role:   {visual, collision}
        // 渲染器根据 role 决定显示内容；导出器忠实描述两者。
        // 详见 ISF_FORMAT_SPEC.md §4.9。

        for (auto& geom : scene.value("geoms", RobotJson::array())) {
            std::shared_ptr<PrimitiveObject> prim;

            // --- 解析 source（GeometryRepresentation）---
            // source 为必填字段：缺失时跳过该 geom，避免直接 [] 访问抛异常终止整场景
            if (!geom.contains("source")) {
                log_warn("RobotNode: geom 缺少必填 source，已跳过 (id={})", geom.value("id", -1));
                continue;
            }
            auto& src = geom["source"];
            std::string kind = src.value("kind", "");
            std::string type = src.value("type", "");
            int typeId = src.value("type_id", -1);
            int dataId = src.value("data_id", -1);
            std::string collRep = src.value("collision_representation", "");
            auto sizes = src.value("size", std::vector<float>{0,0,0});

            if (kind == "mesh" || kind == "sdf") {
                // mesh：外部 OBJ 资源。
                // sdf：MuJoCo 把 SDF geom 关联到一个网格资源（data_id → meshes[]，
                // 导出器输出其表面网格 OBJ；SDF 渲染的零等值面 ≈ 原表面，直接加载 OBJ 即可）。
                auto mit = meshMap.find(dataId);
                if (mit != meshMap.end())
                    prim = makeMesh(mit->second, baseDir);
                else {
                    log_warn("RobotNode: {} data_id={} not found (geom id={})",
                             kind, dataId, geom.value("id", -1));
                    continue;
                }
            } else if (kind == "hfield") {
                // hfield：从高度场数据生成网格（规范 §4.7，data_id → hfields[]）。
                // 内联 data 优先（小数据 JSON 数组，大数据 data_b64 展平 base64）；
                // 均无内联数据时给出提示跳过。
                auto hit = hfieldMap.find(dataId);
                if (hit == hfieldMap.end()) {
                    log_warn("RobotNode: hfield data_id={} not found (geom id={})",
                             dataId, geom.value("id", -1));
                    continue;
                }
                auto& hf = hit->second;
                std::vector<float> hdata;
                if (hf.contains("data") && hf["data"].is_array())
                    hdata = hf["data"].get<std::vector<float>>();
                else if (hf.contains("data_b64") && hf["data_b64"].is_string()) {
                    auto bytes = base64Decode(hf["data_b64"].get<std::string>());
                    if (bytes.size() % sizeof(float) == 0) {
                        hdata.resize(bytes.size() / sizeof(float));
                        std::memcpy(hdata.data(), bytes.data(), bytes.size());
                    }
                }
                prim = makeHField(hf, hdata);
                if (!prim) {
                    log_warn("RobotNode: hfield data 不可用，已跳过 (geom id={})", geom.value("id", -1));
                    continue;
                }
            } else {
                // 基础形状：按类型生成（geom/site 共用逻辑）
                prim = makePrimitiveByType(type, typeId, sizes);
                if (!prim) {
                    log_warn("RobotNode: unsupported primitive type '{}' (id={})",
                             type, geom.value("id",-1));
                    continue;
                }
            }

            if (!prim || prim->verts.size() == 0)
                continue;

            // --- 解析 role（GeometryUsage）---
            bool roleVisual = true;
            bool roleCollision = false;
            if (geom.contains("role")) {
                roleVisual = geom["role"].value("visual", true);
                roleCollision = geom["role"].value("collision", false);
            }

            // 注意：不要将世界变换烘焙到顶点中。
            // mesh 保持在局部空间；世界变换以矩阵形式存储在 userData 上，
            // 供 RobotSceneNode 读取并作为 Matrixes prim 提供给渲染器。

            // ---- 从 initial_global 获取世界变换 ----
            // ISF 1.0：initial_global 包含由 mj_forward 计算的世界空间
            // xpos/xmat，直接使用，不要从 body 层级递归计算。
            glm::mat4 worldMat(1.0f);
            if (geom.contains("initial_global")
                    && geom["initial_global"].contains("xpos")
                    && geom["initial_global"].contains("xmat")) {
                auto& ig = geom["initial_global"];
                auto xpos = ig["xpos"].get<std::vector<float>>();
                auto xmat = ig["xmat"].get<std::vector<float>>();
                worldMat = makeMat4FromXposXmat(xpos, xmat);
            } else if (geom.contains("transform")
                    && geom["transform"].contains("pos")
                    && geom["transform"].contains("quat")) {
                auto gPos = geom["transform"]["pos"].get<std::vector<float>>();
                auto gQuat = geom["transform"]["quat"].get<std::vector<float>>();
                worldMat = makeMat4FromPosQuat(gPos, gQuat);
            } else {
                log_warn("RobotNode: geom 缺少 initial_global/transform 变换，已跳过 (id={})",
                         geom.value("id", -1));
                continue;
            }

            // ---- 在 prim userData 上存 mat_id（绑定由 RobotBindMaterialNode 完成）----
            int geomId = geom.value("id", -1);
            if (geom.contains("material")) {
                int matIdStored = resolveMaterial(geom["material"], geomId,
                                                  matList, matMapOut, sceneKey);
                prim->userData().set2("mat_id", matIdStored);
            }

            // ---- 存储 geom 元数据到 prim userData ----
            prim->userData().set2("geom_name", geom.value("name", std::string("")));
            prim->userData().set2("geom_id", geomId);
            prim->userData().set2("body_id", geom.value("body_id", 0));
            prim->userData().set2("source_kind", kind);
            prim->userData().set2("source_type", type);
            prim->userData().set2("source_data_id", dataId);
            prim->userData().set2("collision_representation", collRep);
            // 可见性由导出端决定：role.visual=false 的 geom 不渲染。
            // 渲染端平台无关，不再按 group 等源引擎特有概念做显隐决策。
            prim->userData().set2("role_visual", roleVisual);
            prim->userData().set2("role_collision", roleCollision);
            // glm::mat4 不能直接用 set2（无 objectFromLiterial 重载），
            // 用 MatrixObject 包装后通过 set 存储。
            auto matObj = std::make_shared<MatrixObject>();
            matObj->m = worldMat;
            prim->userData().set("worldMat", matObj);

            // 暴露 hull 路径用于碰撞可视化
            if (kind == "mesh" && dataId >= 0) {
                auto mit = meshMap.find(dataId);
                if (mit != meshMap.end() && mit->second.contains("hull_obj_path")) {
                    auto hullRel = mit->second["hull_obj_path"];
                    if (!hullRel.is_null()) {
                        auto hullPath = baseDir / std::filesystem::u8path(hullRel.get<std::string>());
                        prim->userData().set2("hull_obj_path", hullPath.string());
                    }
                }
            }

            primList->arr.push_back(prim);
        }

        // ---- 处理 site：作为可渲染图元加入场景（规范 §4.10）----
        // site = 参考标记（reference marker），MuJoCo viewer 默认渲染
        // （如 car.xml 中轮子两侧的绿色 '+' 装饰盒）。可见性由导出端
        // 决策（role.visual，与 geom 一致）；旧数据无 role 时默认显示。
        // 场景键用偏移 id（1000000+site_id）避免与 geom 的 id 冲突。
        if (scene.contains("sites")) {
            for (auto& site : scene["sites"]) {
                std::string sType = site.value("type", std::string("sphere"));
                int sTypeId = site.value("type_id", -1);
                auto sSizes = site.value("size", std::vector<float>{0.01f, 0, 0});
                auto prim = makePrimitiveByType(sType, sTypeId, sSizes);
                if (!prim || prim->verts.size() == 0) {
                    log_warn("RobotNode: unsupported site type '{}' (id={})",
                             sType, site.value("id", -1));
                    continue;
                }

                // 可见性：导出端已按 site 分组决策（--visual-groups，默认 0,1,2）
                bool sRoleVisual = true;
                if (site.contains("role"))
                    sRoleVisual = site["role"].value("visual", true);

                int siteId = 1000000 + site.value("id", -1);
                glm::mat4 worldMat(1.0f);
                if (site.contains("initial_global")
                        && site["initial_global"].contains("xpos")
                        && site["initial_global"].contains("xmat")) {
                    auto& ig = site["initial_global"];
                    worldMat = makeMat4FromXposXmat(
                        ig["xpos"].get<std::vector<float>>(),
                        ig["xmat"].get<std::vector<float>>());
                } else if (site.contains("transform")
                        && site["transform"].contains("pos")
                        && site["transform"].contains("quat")) {
                    worldMat = makeMat4FromPosQuat(
                        site["transform"]["pos"].get<std::vector<float>>(),
                        site["transform"]["quat"].get<std::vector<float>>());
                } else {
                    log_warn("RobotNode: site 缺少 initial_global/transform 变换，已跳过 (id={})",
                             site.value("id", -1));
                    continue;
                }

                if (site.contains("material")) {
                    int matIdStored = resolveMaterial(site["material"], siteId,
                                                      matList, matMapOut, sceneKey);
                    prim->userData().set2("mat_id", matIdStored);
                }

                prim->userData().set2("geom_name", site.value("name", std::string("site")));
                prim->userData().set2("geom_id", siteId);
                prim->userData().set2("body_id", site.value("body_id", 0));
                prim->userData().set2("source_kind", std::string("primitive"));
                prim->userData().set2("source_type", sType);
                prim->userData().set2("source_data_id", -1);
                prim->userData().set2("collision_representation", std::string(""));
                prim->userData().set2("role_visual", sRoleVisual);
                prim->userData().set2("role_collision", false);
                auto matObj = std::make_shared<MatrixObject>();
                matObj->m = worldMat;
                prim->userData().set("worldMat", matObj);

                primList->arr.push_back(prim);
            }
        }

        // ---- 处理 flex：可变形网格（软体/布料，规范 §4.26）----
        // 拓扑静态 + 顶点逐帧更新（动态帧 transforms.flexes 覆盖）。
        // 顶点为世界坐标 → 矩阵恒等；场景节点每帧替换顶点并标记 TotalChange。
        // 场景键用偏移 id（2000000+flex_id）避免与 geom/site 冲突。
        if (scene.contains("flexes")) {
            for (auto& fx : scene["flexes"]) {
                auto prim = std::make_shared<PrimitiveObject>();
                loadFieldVerts(fx, "xpos", prim.get());
                loadFieldTris(fx, "faces", prim.get());
                if (prim->verts.size() == 0 || prim->tris.size() == 0) {
                    log_warn("RobotNode: flex id={} 拓扑为空，已跳过", fx.value("id", -1));
                    continue;
                }
                // per-corner 纹理坐标（跟随 faces，每 3 个对应一个三角形）。
                // JSON 数组 或 texcoords_b64（float32 小端展平，每角点 2 分量）二选一。
                if (fx.contains("texcoords") && fx["texcoords"].is_array()
                        && fx["texcoords"].size() == prim->tris.size() * 3) {
                    auto& uv0 = prim->tris.add_attr<zeno::vec3f>("uv0");
                    auto& uv1 = prim->tris.add_attr<zeno::vec3f>("uv1");
                    auto& uv2 = prim->tris.add_attr<zeno::vec3f>("uv2");
                    uv0.resize(prim->tris.size()); uv1.resize(prim->tris.size());
                    uv2.resize(prim->tris.size());
                    for (size_t t = 0; t < prim->tris.size(); t++) {
                        auto& a = fx["texcoords"][t*3];
                        auto& b = fx["texcoords"][t*3+1];
                        auto& c = fx["texcoords"][t*3+2];
                        uv0[t] = zeno::vec3f(a[0].get<float>(), a[1].get<float>(), 0.0f);
                        uv1[t] = zeno::vec3f(b[0].get<float>(), b[1].get<float>(), 0.0f);
                        uv2[t] = zeno::vec3f(c[0].get<float>(), c[1].get<float>(), 0.0f);
                    }
                    flipUVVertical(prim.get());  // v 方向适配（同 mesh，ISF §3/§4.6.4）
                } else if (fx.contains("texcoords_b64") && fx["texcoords_b64"].is_string()) {
                    auto tbytes = base64Decode(fx["texcoords_b64"].get<std::string>());
                    if (tbytes.size() == prim->tris.size() * 3 * 2 * sizeof(float)) {
                        const float* src = reinterpret_cast<const float*>(tbytes.data());
                        auto& uv0 = prim->tris.add_attr<zeno::vec3f>("uv0");
                        auto& uv1 = prim->tris.add_attr<zeno::vec3f>("uv1");
                        auto& uv2 = prim->tris.add_attr<zeno::vec3f>("uv2");
                        uv0.resize(prim->tris.size()); uv1.resize(prim->tris.size());
                        uv2.resize(prim->tris.size());
                        for (size_t t = 0; t < prim->tris.size(); t++) {
                            uv0[t] = zeno::vec3f(src[t*6],   src[t*6+1], 0.0f);
                            uv1[t] = zeno::vec3f(src[t*6+2], src[t*6+3], 0.0f);
                            uv2[t] = zeno::vec3f(src[t*6+4], src[t*6+5], 0.0f);
                        }
                        flipUVVertical(prim.get());
                    }
                }

                int flexId = fx.value("id", -1);
                int sceneId = 2000000 + flexId;

                // 材质（同 geom：mat_id>=0 查表 / mat_id<0 纯色）
                if (fx.contains("material")) {
                    int matIdStored = resolveMaterial(fx["material"], sceneId,
                                                      matList, matMapOut, sceneKey);
                    prim->userData().set2("mat_id", matIdStored);
                }

                // 可见性：导出端决策（--visual-groups，默认 0,1,2）
                bool fxVisual = true;
                if (fx.contains("role"))
                    fxVisual = fx["role"].value("visual", true);

                prim->userData().set2("geom_name", fx.value("name", std::string("flex")));
                prim->userData().set2("geom_id", sceneId);
                prim->userData().set2("body_id", -1);
                prim->userData().set2("source_kind", std::string("flex"));
                prim->userData().set2("source_type", std::string("flex"));
                prim->userData().set2("source_data_id", flexId);
                prim->userData().set2("collision_representation", std::string(""));
                prim->userData().set2("role_visual", fxVisual);
                prim->userData().set2("role_collision", false);
                prim->userData().set2("is_flex", 1);  // 场景节点据此每帧替换顶点 + TotalChange
                // flex 顶点是世界坐标 → 矩阵恒等
                auto matObj = std::make_shared<MatrixObject>();
                matObj->m = glm::mat4(1.0f);
                prim->userData().set("worldMat", matObj);

                primList->arr.push_back(prim);
            }
        }

        // ---- 处理 skin：蒙皮网格（规范 §4.8）----
        // 拓扑静态（vertices/faces/texcoords 内联）+ 顶点逐帧 LBS 更新
        // （动态帧 transforms.skins[].verts 覆盖，机制同 flex）。
        // 顶点为世界坐标 → 矩阵恒等；场景键 3000000+skin_id。
        if (scene.contains("skins")) {
            for (auto& sk : scene["skins"]) {
                auto prim = std::make_shared<PrimitiveObject>();
                loadFieldVerts(sk, "vertices", prim.get());
                loadFieldTris(sk, "faces", prim.get());
                if (prim->verts.size() == 0 || prim->tris.size() == 0) {
                    log_warn("RobotNode: skin id={} 拓扑为空，已跳过", sk.value("id", -1));
                    continue;
                }
                // skin 的 texcoords 是 per-vertex（每顶点 1 个），转成 per-corner 跟随 faces。
                // JSON 数组 或 texcoords_b64（float32 小端展平，每顶点 2 分量）二选一。
                if (sk.contains("texcoords") && sk["texcoords"].is_array()
                        && sk["texcoords"].size() == prim->verts.size()) {
                    auto& uv0 = prim->tris.add_attr<zeno::vec3f>("uv0");
                    auto& uv1 = prim->tris.add_attr<zeno::vec3f>("uv1");
                    auto& uv2 = prim->tris.add_attr<zeno::vec3f>("uv2");
                    uv0.resize(prim->tris.size()); uv1.resize(prim->tris.size());
                    uv2.resize(prim->tris.size());
                    for (size_t t = 0; t < prim->tris.size(); t++) {
                        auto& i0 = prim->tris[t][0];
                        auto& i1 = prim->tris[t][1];
                        auto& i2 = prim->tris[t][2];
                        auto a = sk["texcoords"][i0], b = sk["texcoords"][i1], c = sk["texcoords"][i2];
                        uv0[t] = zeno::vec3f(a[0].get<float>(), a[1].get<float>(), 0.0f);
                        uv1[t] = zeno::vec3f(b[0].get<float>(), b[1].get<float>(), 0.0f);
                        uv2[t] = zeno::vec3f(c[0].get<float>(), c[1].get<float>(), 0.0f);
                    }
                    flipUVVertical(prim.get());
                } else if (sk.contains("texcoords_b64") && sk["texcoords_b64"].is_string()) {
                    auto tbytes = base64Decode(sk["texcoords_b64"].get<std::string>());
                    if (tbytes.size() == prim->verts.size() * 2 * sizeof(float)) {
                        const float* src = reinterpret_cast<const float*>(tbytes.data());
                        auto& uv0 = prim->tris.add_attr<zeno::vec3f>("uv0");
                        auto& uv1 = prim->tris.add_attr<zeno::vec3f>("uv1");
                        auto& uv2 = prim->tris.add_attr<zeno::vec3f>("uv2");
                        uv0.resize(prim->tris.size()); uv1.resize(prim->tris.size());
                        uv2.resize(prim->tris.size());
                        for (size_t t = 0; t < prim->tris.size(); t++) {
                            auto i0 = prim->tris[t][0], i1 = prim->tris[t][1], i2 = prim->tris[t][2];
                            uv0[t] = zeno::vec3f(src[2*i0],   src[2*i0+1], 0.0f);
                            uv1[t] = zeno::vec3f(src[2*i1],   src[2*i1+1], 0.0f);
                            uv2[t] = zeno::vec3f(src[2*i2],   src[2*i2+1], 0.0f);
                        }
                        flipUVVertical(prim.get());
                    }
                }

                int skinId = sk.value("id", -1);
                int sceneId = 3000000 + skinId;

                // 材质（同 geom/flex：mat_id>=0 查表 / mat_id<0 纯色）
                if (sk.contains("material")) {
                    int matIdStored = resolveMaterial(sk["material"], sceneId,
                                                      matList, matMapOut, sceneKey);
                    prim->userData().set2("mat_id", matIdStored);
                }

                prim->userData().set2("geom_name", sk.value("name", std::string("skin")));
                prim->userData().set2("geom_id", sceneId);
                prim->userData().set2("body_id", -1);
                prim->userData().set2("source_kind", std::string("skin"));
                prim->userData().set2("source_type", std::string("skin"));
                prim->userData().set2("source_data_id", skinId);
                prim->userData().set2("collision_representation", std::string(""));
                prim->userData().set2("role_visual", true);
                prim->userData().set2("role_collision", false);
                prim->userData().set2("is_flex", 1);  // 逐帧替换顶点 + TotalChange（同 flex）
                // skin 顶点是世界坐标 → 矩阵恒等
                auto matObj = std::make_shared<MatrixObject>();
                matObj->m = glm::mat4(1.0f);
                prim->userData().set("worldMat", matObj);

                primList->arr.push_back(prim);
            }
        }
    }

struct RobotStaticNode : zeno::INode {
    virtual void apply() override {
        auto staticPath = get_input2<std::string>("static_path");
        namespace fs = std::filesystem;
        auto staticFilePath = fs::u8path(staticPath);

        std::ifstream file(staticFilePath, std::ios::binary);
        if (!file.is_open()) {
            throw makeError("cannot open static ISF scene file: " + staticPath);
        }
        RobotJson scene;
        file >> scene;
        file.close();

        // 用于解析 mesh/纹理资源的 baseDir（static/ 目录）
        auto baseDir = staticFilePath.parent_path();

        auto primList = std::make_shared<ListObject>();
        auto matMapOut = std::make_shared<RobotMatMap>();
        auto matList = std::make_shared<ListObject>();

        buildStaticScene(scene, baseDir, primList, matList, matMapOut);

        set_output("materials", matList);
        set_output("matMap", matMapOut);
        set_output("prims", primList);
    }
};

ZENDEFNODE(RobotStaticNode, {
    {
        {"readpath", "static_path"},
    },
    {
        {"list", "materials"},
        {"robotmatmap", "matMap"},
        {"object", "prims"},
    },
    {},
    {"robot"},
});

// =====================================================================
// RobotBindMaterialNode：将材质绑定到 prim 上。
//
// 输入：prims（ListObject，来自 RobotStaticNode）和 matMap（RobotMatMap）。
// 遍历每个 PrimitiveObject，从 userData 读取 mat_id，用 matMap 解析为
// mtlidkey，然后设置 matid 属性和 mtlid/matNum/Material_0 userData。
// 不经过此节点的 prim 没有材质绑定 → 渲染器显示白色 mesh。
// =====================================================================
struct RobotBindMaterialNode : zeno::INode {
    virtual void apply() override {
        auto primList = has_input<ListObject>("prims")
            ? get_input<ListObject>("prims") : nullptr;
        auto matMap = has_input<RobotMatMap>("matMap")
            ? get_input<RobotMatMap>("matMap") : nullptr;

        // prims 输入/输出都不含 MaterialObject。
        if (!primList) {
            set_output("prims", std::make_shared<ListObject>());
            return;
        }

        // 遍历 ListObject，对每个 PrimitiveObject 做材质绑定。
        // 渲染器会无条件访问 tris.attr<int>("matid")，因此即使没有材质
        // 也必须添加该属性（避免崩溃）。
        for (auto& obj : primList->arr) {
            auto prim = std::dynamic_pointer_cast<PrimitiveObject>(obj);
            if (!prim) continue;  // 跳过非 PrimitiveObject（如 MaterialObject）

            prim->tris.add_attr<int>("matid");
            prim->tris.attr<int>("matid").assign(prim->tris.size(), 0);

            // 从 prim userData 读取 mat_id，用 matMap 解析为 mtlidkey
            auto& ud = prim->userData();
            int matId = ud.get2<int>("mat_id", -1);
            std::string mtlidkey;
            if (matId >= 0 && matMap) {
                auto it = matMap->matIdToMtlid.find(matId);
                if (it != matMap->matIdToMtlid.end())
                    mtlidkey = it->second;
            }

            if (!mtlidkey.empty()) {
                ud.set2("mtlid", mtlidkey);
                ud.set2("matNum", 1);
                ud.setLiterial("Material_0", mtlidkey);
            }
        }

        set_output("prims", primList);
    }
};

ZENDEFNODE(RobotBindMaterialNode, {
    {
        {"object", "prims"},
        {"robotmatmap", "matMap"},
    },
    {
        {"object", "prims"},
    },
    {},
    {"robot"},
});

// =====================================================================
// RobotDynamicNode：解析动态 ISF manifest（scene_dynamic.isf.json）
// 并将全部动态帧加载到 RobotDynamicData 中。
//
// 结果缓存在此节点实例上：仅当 manifest 路径或其修改时间变化时
// 才重新读取帧。动态辅助函数（manifest/帧加载）被组织为静态方法。
// =====================================================================
struct RobotDynamicNode : zeno::INode {
    std::filesystem::file_time_type m_cachedMtime;
    std::shared_ptr<RobotDynamicData> m_cachedData;

    // ---- 从动态 manifest 路径解析 "dynamic" 目录 ----
    // dynamic_path 指向 scene_dynamic.isf.json；帧文件位于同级的 "frames/" 目录中。
    static std::filesystem::path resolveDynamicDir(const std::filesystem::path& dynamicPath) {
        // dynamic_path = <project>/dynamic/scene_dynamic.isf.json
        // dynamic 目录 = dynamic_path.parent_path()
        return dynamicPath.parent_path();
    }

    // ---- 加载动态 manifest（n_frames、帧文件等）----
    // dynamic_path 指向 scene_dynamic.isf.json（原 manifest.json）。
    // 返回解析后的 manifest JSON（失败时为空）；帧数由公共构建核心从
    // manifest 统一推导。
    static RobotJson loadDynamicManifest(const std::string& dynamicPath) {
        int outNFrames = -1;
        namespace fs = std::filesystem;
        fs::path manifestPath = fs::u8path(dynamicPath);

        std::ifstream file(manifestPath, std::ios::binary);
        if (!file.is_open()) {
            log_warn("RobotNode: cannot open dynamic manifest: {}", manifestPath.string());
            return RobotJson{};
        }
        RobotJson manifest;
        try {
            file >> manifest;
        } catch (const std::exception& e) {
            log_warn("RobotNode: failed to parse manifest: {}", e.what());
            return RobotJson{};
        }
        file.close();

        // manifest 的有序帧列表是权威的。特别是，export_dynamic.py 支持
        // save_every > 1，此时文件是非连续的（例如 frame_000000, _000002）。
        if (manifest.contains("frames") && manifest["frames"].is_array()) {
            outNFrames = static_cast<int>(manifest["frames"].size());
            if (manifest.contains("n_frames") &&
                manifest["n_frames"].get<int>() != outNFrames) {
                log_warn("RobotNode: dynamic manifest n_frames differs from frames list; using frames list");
            }
        } else if (manifest.contains("n_frames")) {
            // 向后兼容的回退方案，用于没有 frames 列表的旧版 manifest，
            // 其文件遵循约定命名。
            outNFrames = manifest["n_frames"].get<int>();
        }
        return manifest;
    }

    // ---- 加载单个动态帧 JSON ----
    // 通过 manifest.frames[].file 解析文件路径。只有不含 frames[] 的
    // 旧版 manifest 才使用旧的 frames/frame_XXXXXX.json 规则。
    static RobotJson loadDynamicFrame(
            const std::string& dynamicPath,
            const RobotJson& manifest,
            int frameIdx) {
        namespace fs = std::filesystem;
        fs::path manifestPath = fs::u8path(dynamicPath);
        fs::path framePath;

        if (manifest.contains("frames") && manifest["frames"].is_array()) {
            if (frameIdx < 0 || frameIdx >= static_cast<int>(manifest["frames"].size())) {
                log_warn("RobotNode: requested frame {} is outside the manifest frame list", frameIdx);
                return RobotJson{};
            }
            const auto& frameMeta = manifest["frames"][frameIdx];
            if (!frameMeta.contains("file") || !frameMeta["file"].is_string()) {
                log_warn("RobotNode: manifest frame {} has no file path", frameIdx);
                return RobotJson{};
            }
            framePath = resolveDynamicDir(manifestPath) /
                fs::u8path(frameMeta["file"].get<std::string>());
        } else {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "frame_%06d.json", frameIdx);
            framePath = resolveDynamicDir(manifestPath) / "frames" / buf;
        }

        std::ifstream file(framePath, std::ios::binary);
        if (!file.is_open()) {
            log_warn("RobotNode: cannot open frame {}: {}", frameIdx, framePath.string());
            return RobotJson{};
        }
        RobotJson frame;
        try {
            file >> frame;
        } catch (const std::exception& e) {
            log_warn("RobotNode: failed to parse frame {}: {}", frameIdx, e.what());
            return RobotJson{};
        }
        file.close();
        return frame;
    }

    static std::filesystem::file_time_type manifestMtime(const std::string& dynamicPath) {
        std::error_code ec;
        return std::filesystem::last_write_time(std::filesystem::u8path(dynamicPath), ec);
    }

    virtual void apply() override {
        auto dynamicPath = get_input2<std::string>("dynamic_path");
        if (dynamicPath.empty()) {
            auto out = std::make_shared<RobotDynamicData>();
            set_output("frames", std::move(out));
            return;
        }

        auto mtime = manifestMtime(dynamicPath);
        if (!m_cachedData || m_cachedData->dynamicPath != dynamicPath || m_cachedMtime != mtime) {
            RobotJson manifest = loadDynamicManifest(dynamicPath);

            auto out = buildDynamicDataFromManifest(
                    manifest, dynamicPath,
                    [&](int i) { return loadDynamicFrame(dynamicPath, manifest, i); });
            m_cachedData = std::move(out);
            m_cachedMtime = mtime;
        }

        set_output("frames", m_cachedData);
    }
};

ZENDEFNODE(RobotDynamicNode, {
    {
        {"readpath", "dynamic_path"},
    },
    {
        {"robotdynamicdata", "frames"},
    },
    {
        {"int", "n_frames", "-1"}
    },
    {"robot"},
});

// =====================================================================
// RobotSceneNode：组装最终的可渲染场景。
//
// 输入：prims（ListObject，来自 RobotBindMaterialNode）、
//       dynamic_frames（RobotDynamicData，来自 RobotDynamicNode）。
// 选取与当前 frameid 匹配的帧，用逐帧变换覆盖静态世界矩阵，
// 通过 SceneAssembler (to_structure()) 构建 scene 列表。
// 材质注册由外部节点单独处理，本节点不再追加 MaterialObject。
// =====================================================================
struct RobotSceneNode : zeno::INode {
    // =====================================================================
    // buildRenderableScene：基于矩阵的动态更新方法。
    //
    // mesh 顶点保持在局部空间（绝不烘焙）。世界变换通过 Matrixes prim
    // 以矩阵形式提供，渲染器通过 OptiX 的 OptixInstance transform 应用它。
    //
    //   第一帧：       mesh = 局部顶点（TotalChange → 上传）
    //                  matrix = W_current（TotalChange → 上传）
    //   后续帧：       mesh = UnChanged（复用 GPU 缓存）
    //                  matrix = W_current（TotalChange → 更新）
    //     → 渲染器：W_current × I × v_local = W_current × v_local
    //
    // to_structure() 生成 DynamicRenderGroups（使用 ROOT 单位矩阵，
    // sub-IAS：无变换）和 DynamicEntries（使用 GEOM 世界矩阵，
    // root-IAS：世界变换），总变换是 W × I × v = W × v（无双重变换）。
    // =====================================================================
    // 递归收集 ListObject / 嵌套 ListObject 里的所有 PrimitiveObject。
    // prims 输入可能来自三种形态，这里统一规整为扁平 prim 列表：
    //   单个 PrimitiveObject、MakeList doConcat=true（平铺 ListObject）、
    //   MakeList doConcat=false（嵌套 ListObject：元素仍是 ListObject）。
    static void collectPrims(
            const std::shared_ptr<IObject>& obj,
            ListObject& out) {
        if (auto prim = std::dynamic_pointer_cast<PrimitiveObject>(obj)) {
            out.arr.push_back(prim);
        } else if (auto lst = std::dynamic_pointer_cast<ListObject>(obj)) {
            for (auto& o : lst->arr)
                collectPrims(o, out);
        }
    }

    // ---- 顶点数据统一解析（动态帧 flex xpos / skin verts）----
    // 兼容两种编码：JSON 嵌套数组（小数据）或 *_b64 字符串（float32 小端展平
    // + base64，大数据，体积约减半）。解析失败（字段缺失/null/长度非法）返回
    // false，out 不变，调用方回退到静态初始顶点。
    static bool parseVertsField(const RobotJson& j, std::vector<std::vector<float>>& out) {
        if (j.is_array()) {
            out = j.get<std::vector<std::vector<float>>>();
            return true;
        }
        if (j.is_string()) {
            auto bytes = base64Decode(j.get<std::string>());
            if (bytes.size() % (3 * sizeof(float)) != 0)
                return false;
            size_t nv = bytes.size() / (3 * sizeof(float));
            const float* src = reinterpret_cast<const float*>(bytes.data());
            out.resize(nv);
            for (size_t i = 0; i < nv; i++)
                out[i] = {src[3*i], src[3*i+1], src[3*i+2]};
            return true;
        }
        return false;
    }

    // 按 (key, key+"_b64") 顺序读取，二选一（导出器只输出其一）。
    static bool readFrameVerts(const RobotJson& frame, const char* key,
                               std::vector<std::vector<float>>& out) {
        if (frame.contains(key))
            return parseVertsField(frame[key], out);
        std::string b64key = std::string(key) + "_b64";
        if (frame.contains(b64key) && frame[b64key].is_string())
            return parseVertsField(frame[b64key], out);
        return false;
    }

    static std::shared_ptr<ListObject> buildRenderableScene(
            const ListObject& primList,
            bool hasDynamic,
            bool isFirstFrame,
            const std::map<int, std::pair<std::vector<float>, std::vector<float>>>& dynGeomXforms,
            const std::map<int, std::vector<std::vector<float>>>& dynFlexVerts
    ) {
        auto sceneObj = std::make_shared<SceneObject>();
        sceneObj->root_name = hasDynamic ? "/RobotScene" : "/RobotStaticScene";
        sceneObj->type = hasDynamic ? "dynamic" : "static";
        sceneObj->matrixMode = hasDynamic ? "TotalChange" : "UnChanged";

        // 根节点：单位矩阵（根位于世界原点）
        SceneTreeNode rootNode;
        rootNode.matrix = sceneObj->root_name + "_m";
        rootNode.visibility = 1;
        sceneObj->node_to_matrix[rootNode.matrix] = {glm::mat4(1)};

        for (auto& obj : primList.arr) {
            auto prim = std::dynamic_pointer_cast<PrimitiveObject>(obj);
            if (!prim) continue;  // 跳过非 PrimitiveObject（如 MaterialObject）

            auto& srcUd = prim->userData();
            std::string geomName = srcUd.get2<std::string>("geom_name", "geom");
            if (geomName.empty()) geomName = "geom";
            int geomId = srcUd.get2<int>("geom_id", -1);
            // 场景键基于 geom_id 生成，避免同名/空名 geom 在
            // scene_tree / prim_list / node_to_matrix 中发生键冲突
            // 导致后面的几何体覆盖前面的（MuJoCo 允许多个 geom 同名）。
            std::string nodeKey = (geomId >= 0)
                ? ("geom_" + std::to_string(geomId))
                : geomName;

            // 从 userData 读取世界变换（静态或动态覆盖）
            // worldMat 以 MatrixObject 形式存储，需手动取出 glm::mat4。
            glm::mat4 worldMat(1.0f);
            if (srcUd.has("worldMat")) {
                auto matObj = std::dynamic_pointer_cast<MatrixObject>(srcUd.get("worldMat"));
                if (matObj) {
                    if (std::holds_alternative<glm::mat4>(matObj->m))
                        worldMat = std::get<glm::mat4>(matObj->m);
                    else if (std::holds_alternative<glm::mat3>(matObj->m))
                        worldMat = glm::mat4(std::get<glm::mat3>(matObj->m));
                }
            }
            if (hasDynamic && geomId >= 0) {
                auto it = dynGeomXforms.find(geomId);
                if (it != dynGeomXforms.end()) {
                    worldMat = makeMat4FromXposXmat(it->second.first, it->second.second);
                }
            }

            std::string nodePath = sceneObj->root_name + "/" + nodeKey;
            std::string matName  = nodePath + "_m";
            std::string meshName = nodeKey + "_mesh";

            // 可见性：由导出端在 role.visual 中决定，渲染端只按它渲染。
            // role.visual=false 的 geom（如碰撞辅助体）不渲染。
            bool roleVisual = srcUd.get2<bool>("role_visual", true);
            int visible = roleVisual ? 1 : 0;

            // 场景树节点
            SceneTreeNode stn;
            stn.meshes    = {meshName};
            stn.matrix    = matName;
            stn.visibility = visible;
            sceneObj->scene_tree[nodePath] = stn;
            if (visible)  // 隐藏节点不加入根 children，避免渲染器实例化
                rootNode.children.push_back(nodePath);

            // 矩阵：世界变换（非单位矩阵，非增量）
            // 渲染器应用：OptixInstance.transform × local_vertex
            sceneObj->node_to_matrix[matName] = {worldMat};

            // Mesh：局部空间顶点（未烘焙）
            // 第一帧：TotalChange（上传到 GPU）
            // 后续帧：UnChanged（复用缓存的 mesh 数据）
            // flex（可变形体）：顶点世界坐标逐帧变化，每帧替换顶点并 TotalChange
            bool isFlex = (srcUd.get2<int>("is_flex", 0) == 1);
            auto scenePrim = std::make_shared<PrimitiveObject>();
            scenePrim->verts = prim->verts;
            scenePrim->tris = prim->tris;
            if (isFlex && geomId >= 0) {
                auto fit = dynFlexVerts.find(geomId);
                if (fit != dynFlexVerts.end() && fit->second.size() == scenePrim->verts.size()) {
                    auto& vs = scenePrim->verts;
                    for (size_t k = 0; k < fit->second.size(); k++) {
                        vs[k] = zeno::vec3f(
                            fit->second[k][0], fit->second[k][1], fit->second[k][2]);
                    }
                }
            }

            auto& ud = scenePrim->userData();
            ud.set2("ObjectName", meshName);
            ud.set2("ResourceType", std::string("Mesh"));
            ud.set2("stamp-change", (isFlex || isFirstFrame)
                    ? std::string("TotalChange")
                    : std::string("UnChanged"));

            // 材质绑定：从源 prim 复制（由 RobotBindMaterialNode 设置）。
            // 如果源 prim 有 matid 属性和 mtlid userData，则复制到场景 prim。
            if (prim->tris.has_attr("matid")) {
                scenePrim->tris.add_attr<int>("matid");
                auto& srcMatid = prim->tris.attr<int>("matid");
                auto& dstMatid = scenePrim->tris.attr<int>("matid");
                dstMatid.assign(srcMatid.begin(), srcMatid.end());
                if (srcUd.has("mtlid")) {
                    ud.set2("mtlid", srcUd.get2<std::string>("mtlid"));
                    ud.set2("matNum", srcUd.get2<int>("matNum"));
                    ud.setLiterial("Material_0", srcUd.get2<std::string>("mtlid"));
                }
            } else {
                // 回退：无 RobotBindMaterialNode 时设置默认 matid
                scenePrim->tris.add_attr<int>("matid");
                scenePrim->tris.attr<int>("matid").assign(scenePrim->tris.size(), 0);
            }

            auto bbox = zeno::primBoundingBox2(scenePrim.get());
            if (bbox.has_value()) {
                const auto& [bmin, bmax] = bbox.value();
                ud.setLiterial("_bboxMin", bmin);
                ud.setLiterial("_bboxMax", bmax);
            }

            sceneObj->prim_list[meshName] = scenePrim;
        }

        sceneObj->scene_tree[sceneObj->root_name] = rootNode;

        // to_structure() 产生：
        //   [mesh prims..., Matrixes prims, SceneDescriptor, SceneTree]
        auto sceneList = sceneObj->to_structure();

        return sceneList;
    }

    virtual void apply() override {
        // prims 输入兼容三种形态：单个 PrimitiveObject、平铺 ListObject、
        // MakeList 嵌套 ListObject（doConcat=false）；统一规整为扁平 prim 列表。
        auto primList = std::make_shared<ListObject>();
        if (has_input("prims"))
            collectPrims(get_input("prims"), *primList);
        auto dynData = has_input<RobotDynamicData>("dynamic_frames")
            ? get_input<RobotDynamicData>("dynamic_frames") : nullptr;

        int nFrames = dynData ? dynData->nFrames : -1;
        bool hasDynamic = dynData && nFrames > 0;

        // 获取当前帧索引
        auto session = &zeno::getSession();
        int currframe = session->globalState->frameid;
        int beginframe = session->globalComm->beginFrameNumber;
        bool isFirstFrame = (currframe == beginframe);

        // 构建动态变换查找表（geom_id → xpos/xmat）
        std::map<int, std::pair<std::vector<float>, std::vector<float>>> dynGeomXforms;
        // 构建 flex 顶点查找表（场景 id=2000000+flex_id → 每帧顶点世界坐标）
        std::map<int, std::vector<std::vector<float>>> dynFlexVerts;
        if (hasDynamic) {
            int frameIdx = currframe - beginframe;
            if (frameIdx < 0) frameIdx = 0;
            if (frameIdx >= nFrames) frameIdx = nFrames - 1;

            const RobotJson& frameData = (frameIdx < (int)dynData->frames.size())
                ? dynData->frames[frameIdx] : dynData->manifest;
            if (frameData.contains("transforms") &&
                frameData["transforms"].contains("geoms")) {
                for (auto& fg : frameData["transforms"]["geoms"]) {
                    int id = fg["id"].get<int>();
                    dynGeomXforms[id] = {
                        fg["xpos"].get<std::vector<float>>(),
                        fg["xmat"].get<std::vector<float>>()
                    };
                }
            }
            // site 变换：静态图元以偏移 id（1000000+site_id）入场景，此处同样偏移匹配
            if (frameData.contains("transforms") &&
                frameData["transforms"].contains("sites")) {
                for (auto& fs : frameData["transforms"]["sites"]) {
                    int id = 1000000 + fs["id"].get<int>();
                    dynGeomXforms[id] = {
                        fs["xpos"].get<std::vector<float>>(),
                        fs["xmat"].get<std::vector<float>>()
                    };
                }
            }
            // flex 顶点（可变形体）：key 同样偏移（2000000+flex_id），逐帧替换顶点。
            // 顶点编码：JSON 数组 或 xpos_b64（float32 展平 base64）二选一。
            if (frameData.contains("transforms") &&
                frameData["transforms"].contains("flexes")) {
                for (auto& ff : frameData["transforms"]["flexes"]) {
                    int id = 2000000 + ff["id"].get<int>();
                    std::vector<std::vector<float>> verts;
                    if (readFrameVerts(ff, "xpos", verts))
                        dynFlexVerts[id] = std::move(verts);
                }
            }
            // skin 顶点（蒙皮 LBS）：key 偏移（3000000+skin_id），逐帧替换顶点（同 flex 机制）
            if (frameData.contains("transforms") &&
                frameData["transforms"].contains("skins")) {
                for (auto& fs : frameData["transforms"]["skins"]) {
                    int id = 3000000 + fs["id"].get<int>();
                    std::vector<std::vector<float>> verts;
                    if (readFrameVerts(fs, "verts", verts))
                        dynFlexVerts[id] = std::move(verts);
                }
            }
        }

        // 构建可渲染场景（可见性由 prim 的 role_visual 决定，导出端已决策）
        auto sceneList = buildRenderableScene(*primList, hasDynamic, isFirstFrame,
                                              dynGeomXforms, dynFlexVerts);
        sceneList->userData().set2("n_frames", nFrames);
        if (hasDynamic && dynData) {
            sceneList->userData().set2("manifest_path", dynData->dynamicPath);
        }

        set_output("scene", std::move(sceneList));
    }
};

ZENDEFNODE(RobotSceneNode, {
    {
        {"object", "prims"},
        {"robotdynamicdata", "dynamic_frames"},
    },
    {
        {"scene"},
    },
    {},
    {"robot"},
});

// =====================================================================
// ReadRobotSimulation：直接从 SequentialPythonNode 的 python 上下文构建场景，
// 取代 RobotStaticNode + RobotDynamicNode 的“从磁盘读中间文件”路径。
//
// 约定：上游 SequentialPythonNode 的脚本完成仿真并把结果写入链私有命名空间：
//   robotStatic   : dict，结构等价于 scene_static.isf.json（meshes/hfields/
//                   materials/textures/geoms/sites/flexes/skins）。
//                   mesh 用内联 vertices/faces，纹理用内联 data_base64，
//                   因此无需磁盘 baseDir。
//   robotDynamic  : dict（可选），结构等价于 scene_dynamic.isf.json：
//                   - 方式一："frames": [ { "transforms": {geoms, sites, flexes,
//                     skins} }, ... ]（每帧完整帧对象）
//                   - 方式二（单帧）：直接含顶层 "transforms"。
//
// 输出与 RobotStaticNode/RobotDynamicNode 一致：materials、matMap、prims、frames。
// =====================================================================
#ifdef ZENO_WITH_PYTHON3

// ---- 将 Python 对象递归转换为 ISF JSON（nlohmann::json） ----
// 支持 bool / int / float / str / list / tuple / dict / None，以及带 `dtype`
// 的 numpy 数组/标量（经 tolist() 规约成嵌套 Python 值）。其余未知类型返回
// null（调用方忽略）。调用前提：已持有 Python GIL。
static RobotJson pyToJson(PyObject* obj) {
    if (!obj || obj == Py_None)
        return nullptr;
    // numpy 数组/标量（float32/int32/bool_ 等并不都是 Python float/int 子类，
    // 需先按类型约定处理）：凡带 `dtype` 属性者用其 tolist() 规约成嵌套
    // Python 列表/标量后再递归转换，避免把 numpy 数据静默转成 null 而下游抛异常。
    if (PyObject_HasAttrString(obj, "dtype") > 0) {
        PyObject* lst = PyObject_CallMethod(obj, "tolist", nullptr);
        if (lst) {
            RobotJson j = pyToJson(lst);
            Py_DECREF(lst);
            return j;
        }
        PyErr_Clear();  // 有 dtype 但无 tolist（罕见类型）：退回常规类型判断
    }
    if (PyBool_Check(obj))
        return (Py_IsTrue(obj) != 0);
    if (PyLong_Check(obj)) {
        long v = PyLong_AsLong(obj);
        if (!PyErr_Occurred())
            return v;
        PyErr_Clear();
        return static_cast<double>(PyLong_AsDouble(obj));
    }
    if (PyFloat_Check(obj))
        return PyFloat_AsDouble(obj);
    if (PyUnicode_Check(obj)) {
        Py_ssize_t n = 0;
        const char* s = PyUnicode_AsUTF8AndSize(obj, &n);
        return s ? RobotJson(std::string(s, (size_t)n)) : RobotJson(nullptr);
    }
    if (PyList_Check(obj) || PyTuple_Check(obj)) {
        RobotJson j = RobotJson::array();
        Py_ssize_t n = PySequence_Size(obj);
        for (Py_ssize_t i = 0; i < n; i++) {
            PyObject* item = PySequence_GetItem(obj, i);
            if (item) {
                j.push_back(pyToJson(item));
                Py_DECREF(item);
            } else {
                PyErr_Clear();
                j.push_back(nullptr);
            }
        }
        return j;
    }
    if (PyDict_Check(obj)) {
        RobotJson j = RobotJson::object();
        PyObject *k = nullptr, *v = nullptr;
        Py_ssize_t pos = 0;
        while (PyDict_Next(obj, &pos, &k, &v)) {
            Py_ssize_t n = 0;
            const char* ks = PyUnicode_AsUTF8AndSize(k, &n);
            if (!ks) {
                PyErr_Clear();
                continue;
            }
            j[std::string(ks, (size_t)n)] = pyToJson(v);
        }
        return j;
    }
    // 无法识别的对象：返回 null（跳过）
    return nullptr;
}

// ---- 轻量结构指纹：探测源数据是否结构性变化 ----
// 仅编码容器类型/长度/字典键序与叶节点标量类型，不读取叶级数值数组，
// 因此遍历整个 robotDynamic 的 300 帧也只是检查框架而非复制大块 xpos/xmat，
// 开销较小。globs 指针相同但结构变化时能据此失效缓存；
// 指针变化（副本重跑新建 dict）由指针比较单独覆盖。
static std::string fingerprintOf(PyObject* obj, int depth) {
    if (depth <= 0)
        return "";
    if (!obj || obj == Py_None)
        return "z";
    if (PyBool_Check(obj))
        return (obj == Py_True) ? "t" : "f";
    if (PyUnicode_Check(obj))
        return "u";
    if (PyLong_Check(obj))
        return "i";
    if (PyFloat_Check(obj))
        return "d";
    if (PyDict_Check(obj)) {
        std::vector<std::string> keys;
        PyObject* k = nullptr;
        PyObject* v = nullptr;
        Py_ssize_t pos = 0;
        while (PyDict_Next(obj, &pos, &k, &v)) {
            const char* ks = PyUnicode_AsUTF8(k);
            if (!ks) {
                PyErr_Clear();
                continue;
            }
            keys.emplace_back(ks);
        }
        std::sort(keys.begin(), keys.end());
        std::string sig = "{";
        for (auto const& kk : keys) {
            PyObject* val = PyDict_GetItemString(obj, kk.c_str());
            sig += kk;
            sig += "=";
            sig += fingerprintOf(val, depth - 1);
            sig += ";";
        }
        sig += "}";
        return sig;
    }
    if (PyList_Check(obj) || PyTuple_Check(obj)) {
        Py_ssize_t n = PySequence_Size(obj);
        std::string sig = "[" + std::to_string(n);
        // 只深入首元素，用其代表整体结构（帧/几何体通常同构）
        if (n > 0) {
            PyObject* first = PySequence_GetItem(obj, 0);
            if (first) {
                sig += ":" + fingerprintOf(first, depth - 1);
                Py_DECREF(first);
            } else {
                PyErr_Clear();
            }
        }
        sig += "]";
        return sig;
    }
    // 其余类型（numpy 数组/标量等在 pyToJson 中处理）仅标记为 opaque
    return "x";
}

// 对链命名空间中 `robotStatic` / `robotDynamic` 两组数据分别取指纹。
// 脚本运行结果的其它变量不参与，避免无关 globals 扰动缓存。
// 调用前提：已持有 Python GIL。
static std::string fingerprintRobots(PyObject* globals) {
    if (!globals)
        return "";
    PyObject* sObj = PyDict_GetItemString(globals, "robotStatic");
    PyObject* dObj = PyDict_GetItemString(globals, "robotDynamic");
    return "S{" + fingerprintOf(sObj, 3) + "}|D{" + fingerprintOf(dObj, 4) + "}";
}

struct ReadRobotSimulation : zeno::INode {
    std::shared_ptr<RobotDynamicData> m_dynData;

    // ---- 缓存：仅当链命名空间（context globals）变化时才重建 ----
    // 调度器每个仿真帧都会重新 apply 本节点（即使源 SequentialPythonNode
    // 标记为 once）。若每帧都重复 pyToJson + buildStaticScene + 重建 300
    // 帧动态数据，会产生大量的重复计算（base64 解码、写临时纹理、网格
    // 重建、整段动态帧序列化），导致 B 方式"特别特别慢"。
    // 与 RobotDynamicNode 按 manifest mtime 缓存同理，这里采用
    // globals 指针 + 结构指纹双键：指针变化（新 dict 重跑）或结构变化
    // （同 dict 原地改写 robotStatic/robotDynamic）都会失效并重建，
    // 既保性能又不至于返回过期数据。m_dynData 在每次未命中时强制清零，
    // 避免上一次上下文残留的 robotDynamic 混入本次静态场景。
    PyObject* m_ctxGlobals = nullptr;
    std::string m_ctxSig;
    std::shared_ptr<ListObject> m_cachedMats;
    std::shared_ptr<RobotMatMap> m_cachedMatMap;
    std::shared_ptr<ListObject> m_cachedPrims;

    virtual void apply() override {
        auto ctx = has_input<PythonContextObject>("robotContext")
            ? get_input<PythonContextObject>("robotContext") : nullptr;

        // Python 对象访问（含取 globals 与结构指纹比对）都需要持有 GIL，
        // 调度器可能在任意工作线程执行本节点。
        PyGILState_STATE gstate = PyGILState_Ensure();
        zeno::scope_exit gstateRelease = [=] { PyGILState_Release(gstate); };

        PyObject* globals = ctx ? ctx->getGlobals() : nullptr;
        std::string ctxSig = globals ? fingerprintRobots(globals) : "";

        // 命中缓存：同一上下文（指针相同）且结构指纹未变，才复用已构建结果。
        if (globals == m_ctxGlobals && ctxSig == m_ctxSig
            && m_cachedMats && m_cachedMatMap && m_cachedPrims) {
            set_output("materials", m_cachedMats);
            set_output("matMap", m_cachedMatMap);
            set_output("prims", m_cachedPrims);
            set_output("frames", m_dynData);
            return;
        }

        // 缓存未命中：先清空上次运行残留的动态数据，避免源数据缺失时
        // 误用上一上下文的帧（修正跨上下文泥入 bug），再重新构建。
        m_dynData.reset();
        auto matMapOut = std::make_shared<RobotMatMap>();
        auto primList = std::make_shared<ListObject>();
        auto matList = std::make_shared<ListObject>();

        if (globals) {
            // 静态信息（结构等价 scene_static.isf.json，资源内联）
            if (PyObject* sObj = PyDict_GetItemString(globals, "robotStatic")) {
                RobotJson scene = pyToJson(sObj);
                if (!scene.is_null() && scene.is_object()) {
                    buildStaticScene(scene, {}, primList, matList, matMapOut);
                }
            }
            // 动态信息（可选；多帧则逐帧变换，单帧则顶层 transforms）
            // 与磁盘路径共用 buildDynamicDataFromManifest：磁盘按 manifest 读
            // 帧文件，这里 loader 直接取内联的 frames[i]，统一解析规则。
            if (PyObject* dObj = PyDict_GetItemString(globals, "robotDynamic")) {
                RobotJson dyn = pyToJson(dObj);
                if (!dyn.is_null() && dyn.is_object()) {
                    m_dynData = buildDynamicDataFromManifest(
                        dyn, std::string("python"),
                        [&](int i) {
                            if (dyn.contains("frames") && dyn["frames"].is_array()
                                    && i < (int)dyn["frames"].size())
                                return dyn["frames"][i];
                            return RobotJson{};  // n_frames 无内联 frames 时安全返回空帧
                        });
                }
            }
        }

        if (!m_dynData)
            m_dynData = std::make_shared<RobotDynamicData>();

        // 缓存本次构建结果（键 = globals 指针 + 结构指纹）。
        m_ctxGlobals = globals;
        m_ctxSig = ctxSig;
        m_cachedMats = matList;
        m_cachedMatMap = matMapOut;
        m_cachedPrims = primList;

        set_output("materials", matList);
        set_output("matMap", matMapOut);
        set_output("prims", primList);
        set_output("frames", m_dynData);
    }
};

ZENDEFNODE(ReadRobotSimulation, {
    {
        {"pythoncontext", "robotContext"},
    },
    {
        {"list", "materials"},
        {"robotmatmap", "matMap"},
        {"object", "prims"},
        {"robotdynamicdata", "frames"},
    },
    {},
    {"robot"},
});

#endif

}
}
