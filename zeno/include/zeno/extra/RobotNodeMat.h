//============== Robot 材质辅助函数（仅头文件）==============
//
// 从 ISF 材质 JSON 构建 MaterialObject，含 PBR frag 着色器生成与纹理加载。
//
// 注意：着色器由 commontpl + common + shadtpl2.first + frag + shadtpl2.second
// 拼接，CallableDefault.cu 模板已把 `cudaTextureObject_t zenotex[]` 声明为
// 函数参数 —— 不要在 common 中再加 `uniform sampler2D zenotex[32]`（冲突导致
// CUDA 编译失败）。

#pragma once

#include <zeno/types/MaterialObject.h>
#include <zeno/types/TextureObject.h>
#include <zeno/utils/vec.h>
#include <zeno/utils/log.h>
#include <tinygltf/json.hpp>
#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace zeno {

// ---- ISF JSON 类型别名 ----
using RobotJson = nlohmann::json;

// ---- 从材质参数构建 PBR frag 着色器字符串 ----
// 模板 (CallableDefault.cu) 引用的所有 mat_* 变量必须在此声明，否则编译报
// "未定义变量"。texIdx 参数：-1 = 该角色无纹理，>=0 = zenotex 索引。
inline std::string buildPBRFragShader(
        const vec4f& baseColor,
        float metallic, float roughness, float specular,
        float emission, float reflectance,
        int diffuseTexIdx,   // -1 = 使用 baseColor，>=0 = 采样 zenotex
        int normalTexIdx,    // -1 = 默认法线 (0,0,1)，>=0 = 采样法线贴图
        int roughnessTexIdx, // -1 = 使用 roughness 数值，>=0 = 采样纹理
        int metallicTexIdx,  // -1 = 使用 metallic 数值，>=0 = 采样纹理
        const vec2f& texRepeat = vec2f(1.0f, 1.0f),  // 2D 纹理 UV 平铺次数（ISF texrepeat）
        bool cubeDiffuse = false,                     // 漫反射槽位是否为 cube 图集（其 UV 不可平铺）
        float transparency = 0.0f,                    // 透明度（0=不透明，1=全透明）
        const std::vector<vec4f>* cubeFaceRects = nullptr  // cube 6 面有效内容矩形（条带内归一化）
) {
    std::ostringstream ss;

    // --- mat_base ---
    ss << "float mat_base = float(1.0);\n";

    // --- mat_basecolor：漫反射纹理或常量 ---
    // texrepeat 仅对 2D 纹理生效；cube 图集 UV 是图集坐标，不能乘平铺次数
    float repX = cubeDiffuse ? 1.0f : texRepeat[0];
    float repY = cubeDiffuse ? 1.0f : texRepeat[1];
    // 漫反射纹理 × 材质颜色调制（ISF §4.5，glTF/Unity/MuJoCo 共同约定）：
    // 最终颜色 = 纹理采样色 × baseColor.rgb。源引擎的"未指定颜色"（如 MuJoCo
    // 编译默认 0.5 灰）由导出器规范化为恒等调制色 [1,1,1]，渲染端无需任何引擎特判。
    ss << "vec3 mat_tint = vec3("
       << baseColor[0] << "," << baseColor[1] << "," << baseColor[2] << ");\n";
    std::string tintMul = " * mat_tint";
    if (diffuseTexIdx >= 0) {
        ss << "vec3 mat_basecolor;\n";   // 先声明，cube/2D 分支各自赋值
        if (cubeDiffuse) {
            // 软件 cube map（ISF §4.4）：cube 纹理按对象空间坐标 (x,y,z) 选面 +
            // 面内坐标采样（等价 GL_TEXTURE_CUBE_MAP 的 GL_OBJECT_LINEAR texgen）。
            // 2D 图集展开在面边界处 UV 跳变，旋转体（capsule/sphere）的图案会断裂；
            // 改为每像素按对象空间位置独立选面 + 面内坐标，跨面图案连续。
            // attrs.objPos = 对象空间 hit 点位置（mesh 保持局部空间，对象空间即局部坐标）。
            ss << "{\n";
            ss << "vec3 mat_dir = vec3(attrs.objPos.x, attrs.objPos.y, attrs.objPos.z);\n";
            ss << "float cdax = fabsf(mat_dir.x), cday = fabsf(mat_dir.y), cdaz = fabsf(mat_dir.z);\n";
            ss << "int cface; float csc, ctc, cma;\n";
            ss << "if (cdax >= cday && cdax >= cdaz) {\n";
            ss << "  if (mat_dir.x >= 0.0f) { cface = 0; csc = -mat_dir.z; ctc = -mat_dir.y; } else { cface = 1; csc = mat_dir.z; ctc = -mat_dir.y; } cma = cdax;\n";
            ss << "} else if (cday >= cdax && cday >= cdaz) {\n";
            ss << "  if (mat_dir.y >= 0.0f) { cface = 2; csc = mat_dir.x; ctc = mat_dir.z; } else { cface = 3; csc = mat_dir.x; ctc = -mat_dir.z; } cma = cday;\n";
            ss << "} else {\n";
            ss << "  if (mat_dir.z >= 0.0f) { cface = 4; csc = mat_dir.x; ctc = -mat_dir.y; } else { cface = 5; csc = -mat_dir.x; ctc = -mat_dir.y; } cma = cdaz;\n";
            ss << "}\n";
            ss << "float ccs = (csc / cma + 1.0f) / 2.0f;\n";
            ss << "float cct = (ctc / cma + 1.0f) / 2.0f;\n";
            if (cubeFaceRects && cubeFaceRects->size() == 6) {
                // 条带内有效内容矩形（导出器检测），避免采样到黑边
                ss << "float cfr0[6] = {";
                for (int f = 0; f < 6; f++) ss << (f ? ", " : "") << (*cubeFaceRects)[f][0];
                ss << "};\n";
                ss << "float cfr1[6] = {";
                for (int f = 0; f < 6; f++) ss << (f ? ", " : "") << (*cubeFaceRects)[f][1];
                ss << "};\n";
                ss << "float cfr2[6] = {";
                for (int f = 0; f < 6; f++) ss << (f ? ", " : "") << (*cubeFaceRects)[f][2];
                ss << "};\n";
                ss << "float cfr3[6] = {";
                for (int f = 0; f < 6; f++) ss << (f ? ", " : "") << (*cubeFaceRects)[f][3];
                ss << "};\n";
                ss << "float ccu = cfr0[cface] + ccs * (cfr2[cface] - cfr0[cface]);\n";
                ss << "float ccv = 1.0f - (float(cface) + cfr1[cface] + cct * (cfr3[cface] - cfr1[cface])) / 6.0f;\n";
            } else {
                ss << "float ccu = ccs;\n";
                ss << "float ccv = 1.0f - (float(cface) + cct) / 6.0f;\n";
            }
            ss << "mat_basecolor = vec3(texture2D(zenotex["
               << diffuseTexIdx << "], vec2(ccu, ccv)))" << tintMul << ";\n";
            ss << "}\n";
        } else {
            ss << "mat_basecolor = vec3(texture2D(zenotex["
               << diffuseTexIdx << "], vec2(att_uv) * vec2("
               << repX << "," << repY << ")))" << tintMul << ";\n";
        }
    } else {
        ss << "vec3 mat_basecolor = vec3("
           << baseColor[0] << "," << baseColor[1] << "," << baseColor[2] << ");\n";
    }

    // --- mat_roughness：粗糙度纹理或常量 ---
    if (roughnessTexIdx >= 0) {
        ss << "float mat_roughness = float(texture2D(zenotex["
           << roughnessTexIdx << "], vec2(att_uv) * vec2("
           << texRepeat[0] << "," << texRepeat[1] << ")).x);\n";
    } else {
        ss << "float mat_roughness = float(" << roughness << ");\n";
    }

    // --- mat_metallic：金属度纹理或常量 ---
    if (metallicTexIdx >= 0) {
        ss << "float mat_metallic = float(texture2D(zenotex["
           << metallicTexIdx << "], vec2(att_uv) * vec2("
           << texRepeat[0] << "," << texRepeat[1] << ")).x);\n";
    } else {
        ss << "float mat_metallic = float(" << metallic << ");\n";
    }

    // --- 剩余 mat_* 变量（必须全部声明）---
    ss << "vec3 mat_metalColor = vec3(1.0,1.0,1.0);\n";
    ss << "float mat_specular = float(" << specular << ");\n";
    ss << "float mat_specularTint = float(0.0);\n";
    ss << "float mat_anisotropic = float(0.0);\n";
    ss << "float mat_anisoRotation = float(0.0);\n";
    ss << "float mat_subsurface = float(0.0);\n";
    ss << "vec3 mat_sssParam = vec3(1.0,1.0,1.0);\n";
    ss << "float mat_sssScale = float(1.0);\n";
    ss << "vec3 mat_sssColor = vec3(1.0,1.0,1.0);\n";
    ss << "float mat_scatterDistance = float(10000.0);\n";
    ss << "float mat_scatterStep = float(0.0);\n";
    ss << "float mat_sheen = float(0.0);\n";
    ss << "float mat_sheenTint = float(0.5);\n";
    ss << "float mat_clearcoat = float(0.0);\n";
    ss << "vec3 mat_clearcoatColor = vec3(1.0,1.0,1.0);\n";
    ss << "float mat_clearcoatRoughness = float(0.0);\n";
    ss << "float mat_clearcoatIOR = float(1.5);\n";
    ss << "float mat_specTrans = float(0.0);\n";
    ss << "float mat_F0 = float(0.04);\n";
    ss << "vec3 mat_transColor = vec3(1.0,1.0,1.0);\n";
    ss << "vec3 mat_transTint = vec3(1.0,1.0,1.0);\n";
    ss << "float mat_transTintDepth = float(10000.0);\n";
    ss << "float mat_transDistance = float(1.0);\n";
    ss << "vec3 mat_transScatterColor = vec3(1.0,1.0,1.0);\n";
    ss << "float mat_ior = float(1.5);\n";
    ss << "float mat_diffraction = float(0.0);\n";
    ss << "vec3 mat_diffractColor = vec3(0.0,0.0,0.0);\n";
    ss << "float mat_flatness = float(0.0);\n";
    ss << "float mat_shadowReceiver = float(0.0);\n";
    ss << "float mat_shadowTerminatorOffset = float(0.0);\n";
    ss << "float mat_thin = float(0.0);\n";
    ss << "float mat_doubleSide = float(0.0);\n";

    // --- mat_normal：法线贴图纹理或默认值 (0,0,1) ---
    // 注意：texture2D() 返回 vec4（zxxglslvec.h），它没有 `.xyz` 成员 ——
    // 必须使用显式 vec3(vec4) 构造，否则 CUDA 着色器编译失败，
    // 渲染器会静默回退到默认（白色）材质。
    if (normalTexIdx >= 0) {
        ss << "vec3 mat_normal = normalize(vec3(texture2D(zenotex["
           << normalTexIdx << "], vec2(att_uv) * vec2("
           << texRepeat[0] << "," << texRepeat[1] << "))) * 2.0 - 1.0);\n";
    } else {
        ss << "vec3 mat_normal = vec3(0,0,1);\n";
    }

    ss << "float mat_displacement = float(0.0);\n";
    ss << "float mat_smoothness = float(1.0);\n";
    ss << "float mat_emissionIntensity = float(" << emission << ");\n";
    // 渲染器按 mat_emissionIntensity * mat_emission 组装自发光（CallableDefault.cu），
    // 因此强度放 intensity，颜色保持白色，避免发光被二次放大（emission²）。
    ss << "vec3 mat_emission = vec3(1.0,1.0,1.0);\n";
    ss << "float mat_emissionOnly = float(0.0);\n";
    ss << "vec3 mat_reflectance = vec3(" << reflectance << "," << reflectance << "," << reflectance << ");\n";
    // mat_opacity 在渲染器中按“透明度”解释（0=不透明，1=全透明，见 DeflMatShader.cu）
    ss << "float mat_opacity = float(" << transparency << ");\n";
    ss << "float mat_thickness = float(0.0);\n";
    ss << "float mat_isHair = float(0.0);\n";
    ss << "float mat_HairRough2 = float(1.0);\n";
    ss << "float mat_FurCoat = float(0.0);\n";
    ss << "bool sssFxiedRadius = true;\n";
    ss << "vec3 mask_value = vec3(0,0,0);\n";

    return ss.str();
}

// ---- 从 ISF 纹理路径创建 Texture2DObject ----
inline std::shared_ptr<Texture2DObject> makeTexture2D(const std::filesystem::path& texPath) {
    auto tex = std::make_shared<Texture2DObject>();
    tex->path = texPath.string();
    tex->wrapS = Texture2DObject::TexWrapEnum::REPEAT;
    tex->wrapT = Texture2DObject::TexWrapEnum::REPEAT;
    tex->minFilter = Texture2DObject::TexFilterEnum::LINEAR_MIPMAP_LINEAR;
    tex->magFilter = Texture2DObject::TexFilterEnum::LINEAR;
    tex->blockCompression = false;
    return tex;
}

// ---- base64 解码（RFC 4648 标准表）----
// 供全内联场景（ReadRobotSimulation）使用：纹理以 data_base64 内嵌于 ISF dict，
// 渲染前解码为字节。
inline std::vector<unsigned char> base64Decode(const std::string& in) {
    static const char* b64tbl = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    int tbl[256];
    for (int i = 0; i < 256; i++) tbl[i] = -1;
    for (int i = 0; i < 64; i++) tbl[(unsigned char)b64tbl[i]] = i;
    std::vector<unsigned char> out;
    int val = 0, bits = -8;
    for (unsigned char c : in) {
        if (c == '=') break;
        int d = tbl[c];
        if (d < 0) continue;  // 忽略空白/非法字符
        val = (val << 6) | d;
        bits += 6;
        if (bits >= 0) {
            out.push_back((unsigned char)((val >> bits) & 0xFF));
            bits -= 8;
        }
    }
    return out;
}

// ---- 内容哈希（FNV-1a 64bit → 16 位 hex）----
// 用于内联纹理的稳定文件名。无状态实现，线程安全。
inline std::string contentHash(const std::string& s) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ull;
    }
    static const char* hexd = "0123456789abcdef";
    std::string out(16, '0');
    for (int i = 15; i >= 0; --i) {
        out[i] = hexd[h & 0xF];
        h >>= 4;
    }
    return out;
}

// ---- 将内联 base64 字节落为临时 PNG 供渲染器加载 ----
// Texture2DObject 仅携带 path（渲染器按路径加载），因此把内嵌字节物化到
// 临时目录。对数据契约而言仍是"内联"（无场景级外部文件依赖）。
// 文件名 = 内容哈希：同一纹理内容恒映射同一路径，渲染器按路径缓存时跨场景
// 天然复用且互不污染；不同内容必然不同路径，杜绝顺序编号导致的跨场景路径
// 漂移（新场景误加载旧场景残留纹理）。
inline std::filesystem::path materializeInlineTexture(const std::string& base64str) {
    auto bytes = base64Decode(base64str);
    auto path = std::filesystem::temp_directory_path() /
        ("zeno_inline_tex_" + contentHash(base64str) + ".png");
    std::error_code ec;
    if (std::filesystem::exists(path, ec))
        return path;  // 同内容已物化，直接复用
    std::ofstream ofs(path, std::ios::binary);
    if (ofs && !bytes.empty())
        ofs.write((const char*)bytes.data(), (std::streamsize)bytes.size());
    return path;
}

// ---- 从 ISF 材质 JSON 构建 MaterialObject ----
// ISF role → tex2Ds 索引映射：纹理按顺序添加到 tex2Ds，
// tex2Ds 中的索引对应着色器中的 zenotex[N] 槽位。
inline std::shared_ptr<MaterialObject> buildMaterialObject(
        const RobotJson& matJson,
        const std::map<int, RobotJson>& texMap,
        const std::filesystem::path& baseDir,
        const std::string& mtlidkey
) {
    auto mtl = std::make_shared<MaterialObject>();
    mtl->mtlidkey = mtlidkey;

    // ---- 解析 PBR 参数（ISF -1 表示"未定义"，使用默认值）----
    vec4f baseColor(1.0f, 1.0f, 1.0f, 1.0f);
    float metallic = 0.0f;
    float roughness = 0.5f;
    float specular = 0.5f;
    float emission = 0.0f;
    float reflectance = 0.0f;

    if (matJson.contains("rgba")) {
        auto rgba = matJson["rgba"].get<std::vector<float>>();
        if (rgba.size() >= 4)
            baseColor = vec4f(rgba[0], rgba[1], rgba[2], rgba[3]);
    }
    if (matJson.contains("pbr")) {
        auto& pbr = matJson["pbr"];
        if (pbr.contains("base_color")) {
            auto bc = pbr["base_color"].get<std::vector<float>>();
            if (bc.size() >= 4)
                baseColor = vec4f(bc[0], bc[1], bc[2], bc[3]);
        }
        float m = pbr.value("metallic", -1.0f);
        metallic = (m < 0) ? 0.0f : m;          // -1 → 默认 0
        float r = pbr.value("roughness", -1.0f);
        roughness = (r < 0) ? 0.5f : r;          // -1 → 默认 0.5
        float s = pbr.value("specular", -1.0f);
        specular = (s < 0) ? 0.5f : s;           // -1 → 默认 0.5
        float e = pbr.value("emission", -1.0f);
        emission = (e < 0) ? 0.0f : e;           // -1 → 默认 0.0
        float rf = pbr.value("reflectance", -1.0f);
        reflectance = (rf < 0) ? 0.0f : rf;       // -1 → 默认 0.0
    }

    // rgba 的 alpha 分量决定材质透明度（ISF 规范 4.5 节：alpha<1 表示半透明）
    float alpha = std::clamp(baseColor[3], 0.0f, 1.0f);

    // ISF texrepeat（UV 平铺次数，仅对 2D 纹理生效，见 ISF 规范 4.5 节）
    vec2f texRepeat(1.0f, 1.0f);
    if (matJson.contains("texrepeat")) {
        auto tr = matJson["texrepeat"].get<std::vector<float>>();
        if (tr.size() >= 2)
            texRepeat = vec2f(tr[0], tr[1]);
    }

    // ---- 按角色收集纹理，分配 zenotex 索引 ----
    // 注意：MuJoCo cube 纹理 (type:"cube") 始终是颜色纹理。
    // 导出器将它们标记为 role:"normal"，但它们实际上是
    // 漫反射/颜色纹理。无论 role 标签如何，都将 type:"cube" 视为漫反射。
    int diffuseTexIdx = -1, normalTexIdx = -1;
    int roughnessTexIdx = -1, metallicTexIdx = -1;
    bool cubeDiffuse = false;
    // cube 纹理每面的有效内容矩形（条带内归一化，导出器检测，用于图集 UV 映射）
    std::vector<vec4f> cubeFaceRects;

    if (matJson.contains("textures")) {
        for (auto& mt : matJson["textures"]) {
            int texId = mt.value("tex_id", -1);
            std::string role = mt.value("role", std::string(""));
            auto tit = texMap.find(texId);
            if (tit == texMap.end())
                continue;

            // 纹理数据来源：内联 data_base64（全内联场景）优先，否则外部 data_path。
            bool hasInline = tit->second.contains("data_base64") && tit->second["data_base64"].is_string();
            bool hasPath = tit->second.contains("data_path");
            if (!hasInline && !hasPath)
                continue;

            // 检查纹理类型 —— cube 纹理始终是漫反射（颜色）
            std::string texType = tit->second.value("type", std::string(""));

            std::shared_ptr<Texture2DObject> tex;
            if (hasInline) {
                auto tmpPath = materializeInlineTexture(tit->second["data_base64"].get<std::string>());
                tex = makeTexture2D(tmpPath);
            } else {
                auto texPath = baseDir / std::filesystem::u8path(
                    tit->second["data_path"].get<std::string>());
                // 用 error_code 版本避免 IO 错误（权限/路径异常）时抛出 filesystem_error
                std::error_code ec;
                if (!std::filesystem::exists(texPath, ec)) {
                    log_warn("RobotNode: 纹理文件不存在: {}", texPath.string());
                }
                tex = makeTexture2D(texPath);
            }
            if (!tex)
                continue;

            int idx = static_cast<int>(mtl->tex2Ds.size());
            mtl->tex2Ds.push_back(tex);

            // ISF role → 着色器纹理槽位
            // （cube 纹理始终作为漫反射处理）
            if (texType == "cube") {
                // MuJoCo cube 纹理是颜色纹理，不是法线贴图
                diffuseTexIdx = idx;
                cubeDiffuse = true;
                // 读取导出器检测的面内容矩形（条带内归一化）
                if (tit->second.contains("face_rects")
                        && tit->second["face_rects"].is_array()
                        && tit->second["face_rects"].size() == 6) {
                    cubeFaceRects.clear();
                    for (auto& fr : tit->second["face_rects"])
                        cubeFaceRects.emplace_back(
                            fr[0].get<float>(), fr[1].get<float>(),
                            fr[2].get<float>(), fr[3].get<float>());
                }
            } else if (role == "diffuse" || role == "rgba") {
                diffuseTexIdx = idx;
            } else if (role == "normal") {
                normalTexIdx = idx;
            } else if (role == "roughness") {
                roughnessTexIdx = idx;
            } else if (role == "metallic") {
                metallicTexIdx = idx;
            } else {
                // occlusion/emission/specular/glossiness/bump 等角色
                // 当前 PBR 着色器未使用，给出提示便于排查
                log_warn("RobotNode: 未支持的纹理角色 '{}'（tex_id={}），已跳过", role, texId);
            }
        }
    }

    // ---- 重要：不要在 common 中添加 `uniform sampler2D zenotex[32]`！----
    // CallableDefault.cu 模板已将 zenotex 声明为函数参数：
    // `cudaTextureObject_t zenotex[]`。添加文件作用域的
    // `uniform sampler2D zenotex[32]` 会与之冲突（sampler2D 和
    // uniform 是 GLSL 关键字，在 CUDA 中无效），导致着色器编译
    // 失败并静默回退到默认材质。
    //
    // mtl->common 保持为空 —— 模板已提供所需的一切。

    // ---- frag：包含所有 mat_* 变量的 PBR 着色器 ----
    mtl->frag = buildPBRFragShader(
        baseColor, metallic, roughness, specular, emission, reflectance,
        diffuseTexIdx, normalTexIdx, roughnessTexIdx, metallicTexIdx,
        texRepeat, cubeDiffuse, 1.0f - alpha,
        cubeFaceRects.empty() ? nullptr : &cubeFaceRects);

    // ---- parameters JSON（opacity=2.0 表示不透明，见 ShaderFinalize.cpp 惯例）----
    // alpha<1 时按 alpha 输出，frag 中的 mat_opacity 走随机透明度测试
    nlohmann::json j;
    j["opacity"] = (alpha < 1.0f) ? alpha : 2.0;
    mtl->parameters = j.dump();

    return mtl;
}

// ---- 构建简单纯色 MaterialObject（用于 mat_id=-1 的 geom）----
inline std::shared_ptr<MaterialObject> buildSolidColorMaterial(
        const vec4f& rgba, const std::string& mtlidkey
) {
    auto mtl = std::make_shared<MaterialObject>();
    mtl->mtlidkey = mtlidkey;
    float alpha = std::clamp(rgba[3], 0.0f, 1.0f);
    mtl->frag = buildPBRFragShader(
        rgba, 0.0f, 0.5f, 0.5f, 0.0f, 0.0f,
        -1, -1, -1, -1,  // 无纹理
        vec2f(1.0f, 1.0f), false, 1.0f - alpha);
    nlohmann::json j;
    j["opacity"] = (alpha < 1.0f) ? alpha : 2.0;
    mtl->parameters = j.dump();
    return mtl;
}

} // namespace zeno
