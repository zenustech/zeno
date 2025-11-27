#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/types/StringObject.h>
#include <zeno/utils/string.h>
#include <zeno/utils/fileio.h>
#include <zeno/utils/logger.h>
#include <zeno/utils/vec.h>
#include <string_view>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <cstdio>
#include <fstream>
#include <zeno/types/UserData.h>

#include <tinygltf/json.hpp>
#include "zeno/extra/TempNode.h"

using Json = nlohmann::json;

namespace zeno {
struct JsonObject : IObjectClone<JsonObject> {
    Json json;
};
namespace {

template <std::size_t ...Is>
static bool match_helper(char const *&it, char const *arr, std::index_sequence<Is...>) {
    if (((it[Is] == arr[Is]) && ...)) {
        it += sizeof...(Is);
        return true;
    } else {
        return false;
    }
}

template <std::size_t N>
static bool match(char const *&it, char const (&arr)[N]) {
    return match_helper(it, arr, std::make_index_sequence<N - 1>{});
}

static float takef(char const *&it) {
    char *eptr;
    float val = std::strtof(it, &eptr);
    it = eptr;
    return val;
}
static std::string takes(std::string &s_in)
{

    char c = s_in.c_str()[0];
    std::string s; s = c;
    return s;
}
static int takeu(char const *&it) {
    char *eptr;
    int val(std::strtoul(it, &eptr, 10));
    it = eptr;
    return val;
}

// std::shared_ptr<PrimitiveObject> parse_obj(std::vector<char> &&bin) 
PrimitiveObject* parse_obj(const char *binData, std::size_t binSize) {
    /*bin.resize(bin.size() + 8, '\0');*/

    char const *it = binData;
    char const *eit = binData + binSize;// - 8;

    // auto prim = std::make_shared<PrimitiveObject>();
    auto prim = new PrimitiveObject;
    prim->polys.add_attr<int>("matid");
    std::vector<int> loop_uvs;
    int mat_num = 0;
    int prev_poly = 0;
    int current_mat_id = 0;
    while (it < eit) {
        auto nit = std::find(it, eit, '\n');
        auto nnit = nit + 1;
        if (nit[-1] == '\r')
            --nit;

        if (match(it, "v ")) {
            float x = takef(it);
            float y = takef(it);
            float z = takef(it);
            prim->verts.emplace_back(x, y, z);

        } else if (match(it, "vt ")) {
            float x = takef(it);
            float y = takef(it);
            prim->uvs.emplace_back(x, y);

        } else if (match(it, "f ")) {
            int beg = prim->loops.size();
            int cnt{};
            while (it != nit) {
                int x = takeu(it) - 1;
                if (*it == '/' && it[1] != '/') {
                    ++it;
                    int xt = takeu(it) - 1;
                    loop_uvs.push_back(xt);
                }
                it = std::find(it, nit, ' ');
                prim->loops.push_back(x);
                ++cnt;
                it = std::find_if(it, nit, [] (char c) { return c != ' '; });
            }
            prim->polys.emplace_back(beg, cnt);
            if(prim->polys.has_attr("matid")) {
                prim->polys.attr<int>("matid").emplace_back(current_mat_id);
            }

        } else if (match(it, "l ")) {
            int x = takeu(it) - 1;
            int y = takeu(it) - 1;
            prim->lines.emplace_back(x, y);

        //} else if (match(it, "o ")) {
            // todo: support tag verts to be multi components of primitive
            //std::string_view o_name(it, nit - it);

        }else if (match(it, "usemtl "))
        {
//            std::string s(it);
            std::string mat_name = std::string(it, nit-it);
//            printf("%s\n", mat_name.c_str());
            current_mat_id = mat_num;
            auto matkey = "Material_"+std::to_string(mat_num);
            prim->userData().set2(matkey, mat_name);
            mat_num++;
            prim->userData().set2("matNum", mat_num);
        }
        it = nnit;
    }

    {
        int vert_count = prim->verts.size();
        for (auto &i : prim->loops) {
            if (i < 0) {
                i += vert_count + 1;
            }
        }
        for (auto &i : loop_uvs) {
            if (i < 0) {
                i += vert_count + 1;
            }
        }
    }

    if (loop_uvs.size() == prim->loops.size()) {
        prim->loops.add_attr<int>("uvs") = std::move(loop_uvs);
    }

    return prim;
}
/**
 * @brief 将字符串转换为小写。
 * @param str 要转换的字符串。
 * @return 小写字符串。
 */
static std::string to_lower(std::string str) {
    // 使用 std::transform 和 std::tolower 进行高效转换
    std::transform(str.begin(), str.end(), str.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return str;
}

/**
 * @brief 从一个分割后的字符串列表中解析纹理选项和实际路径。
 * * MTL格式允许纹理参数（如 -s 1 1 1）出现在文件路径之前或之后。
 * * @param parts 纹理定义行分割后的字符串列表。
 * @return 包含 (实际纹理路径, 选项字典) 的 pair。
 */
static std::pair<std::string, Json> parse_texture_options_and_path(const std::vector<std::string>& parts) {
    Json options = Json::object();
    std::string actual_path;

    if (parts.empty()) {
        return {actual_path, options};
    }

    // 确定实际路径：通常是最后一个不是选项（不以'-'开头）的字符串
    std::vector<std::string> path_candidates;
    for (const auto& part : parts) {
        if (!part.empty() && part[0] != '-') {
            path_candidates.push_back(part);
        }
    }

    if (!path_candidates.empty()) {
        actual_path = path_candidates.back();
    }

    // 重新处理所有 parts 来解析选项
    for (size_t i = 0; i < parts.size(); ++i) {
        const std::string& opt = parts[i];
        if (opt.size() > 1 && opt[0] == '-') {
            std::string opt_name = to_lower(opt.substr(1));

            // 检查下一个部分是否是选项的值
            if (i + 1 < parts.size()) {
                const std::string& next_part = parts[i + 1];
                // 如果下一个部分不是另一个选项且不是实际路径
                if (next_part[0] != '-' && next_part != actual_path) {
                    options[opt_name] = next_part;
                    i++; // 跳过下一个值
                } else {
                    options[opt_name] = true; // 标记选项存在
                }
            } else {
                options[opt_name] = true;
            }
        }
    }

    return {actual_path, options};
}

/**
 * @brief 将MTL材质字符串内容解析为结构化的JSON数据。
 * @param mtl_content 包含完整MTL文件内容的字符串。
 * @return 包含所有材质数据的 JSON 对象。
 */

static std::string resolve_tex_path(std::string tex_path, std::string HintDirectory) {
    auto outs = zeno::TempNodeSimpleCaller("ResolveTexPath")
            .set2<std::string>("tex_path", tex_path)
            .set2<std::string>("HintDirectory", HintDirectory)
            .call();
    auto real_path = outs.get2<std::string>("real_path");
    return real_path;
}
static Json convert_mtl_to_json(const std::string& mtl_content, std::string hint_dir) {
    Json materials = Json::object();
    std::string current_material_name; // 使用空字符串作为 "无当前材质" 的标志

    // 使用 stringstream 处理输入的字符串内容
    std::stringstream ss_content(mtl_content);

    std::string line;
    int line_number = 0;

    // 循环读取 stringstream 中的每一行
    while (std::getline(ss_content, line)) {
        line_number++;

        // 移除行首尾的空格
        size_t start = line.find_first_not_of(" \t");
        if (start == std::string::npos) continue;
        line = line.substr(start);

        // 跳过注释
        if (line.empty() || line.rfind('#', 0) == 0) continue;

        std::stringstream ss_line(line);
        std::string token;
        std::vector<std::string> parts;
        while (ss_line >> token) {
            parts.push_back(token);
        }

        if (parts.empty()) continue;

        std::string key = to_lower(parts[0]);

        // 新建材质 (newmtl)
        if (key == "newmtl" && parts.size() > 1) {
            current_material_name = parts[1];
            // 初始化材质属性，提供默认值
            Json material = {
                    {"name", current_material_name},
                    {"ambient_value", {0.2, 0.2, 0.2}},          // Ka
                    {"diffuse_value", {0.8, 0.8, 0.8}},          // Kd
                    {"specular_value", {1.0, 1.0, 1.0}},         // Ks
                    {"emissive_value", {0.0, 0.0, 0.0}},         // Ke
                    {"transmission_filter", {1.0, 1.0, 1.0}}, // Tf
                    {"shininess_value", 32.0},                   // Ns
                    {"optical_density", 1.0},              // Ni
                    {"opacity_value", 0.0},                // d
                    {"transparency", 1.0},                 // Tr
                    {"illumination_model", 2},             // illum
                    {"ambient_tex", ""},                   // map_Ka
                    {"diffuse_tex", ""},                   // map_Kd
                    {"specular_tex", ""},                  // map_Ks
                    {"emissive_tex", ""},                  // map_Ke
                    {"shininess_tex", ""},                 // map_Ns
                    {"opacity_tex", ""},                   // map_d
                    {"bump_tex", ""},                      // map_Bump, bump
                    {"displacement_tex", ""},              // disp
                    {"decal_tex", ""},                     // decal
                    {"reflection_tex", ""},                // refl
                    {"normal_map_tex", ""},                // norm
                    {"texture_options", Json::object()}
            };
            materials[current_material_name] = material;
        }

        // 处理材质属性
        else if (!current_material_name.empty() && materials.count(current_material_name)) {
            Json& material = materials[current_material_name];

            try {
                if (key == "ka" || key == "kd" || key == "ks" || key == "ke" || key == "tf") {
                    if (parts.size() < 4) {
                        throw std::runtime_error("color value component less than 3");
                    }
                    Json values = {std::stod(parts[1]), std::stod(parts[2]), std::stod(parts[3])};
                    if (key == "ka") material["ambient_value"] = values;
                    else if (key == "kd") material["diffuse_value"] = values;
                    else if (key == "ks") material["specular_value"] = values;
                    else if (key == "ke") material["emissive_value"] = values;
                    else if (key == "tf") material["transmission_filter"] = values;
                }
                else if (key == "ns" || key == "ni" || key == "d" || key == "tr" || key == "illum") {
                    if (parts.size() < 2) {
                        throw std::runtime_error("missing params");
                    }
                    if (key == "ns") material["shininess_value"] = std::stod(parts[1]);
                    else if (key == "ni") material["optical_density"] = std::stod(parts[1]);
                    else if (key == "d") material["opacity_value"] = std::stod(parts[1]);
                    else if (key == "tr") material["transparency"] = std::stod(parts[1]); // Tr 是透明度的反向
                    else if (key == "illum") material["illumination_model"] = std::stoi(parts[1]);
                }
                    // 纹理映射
                else if (key.rfind("map_", 0) == 0 || key == "bump" || key == "disp" || key == "decal" || key == "refl" || key == "norm") {
                    // 提取纹理行中的参数（跳过 key）
                    std::vector<std::string> texture_parts(parts.begin() + 1, parts.end());

                    // 已经使用了 C++17 结构化绑定
                    auto [texture_path, options] = parse_texture_options_and_path(texture_parts);

                    texture_path = zeno::replace_all(texture_path, "\\", "/");
                    texture_path = resolve_tex_path(texture_path, hint_dir);

                    // 映射到材质字典中的对应键
                    if (key == "map_ka") material["ambient_tex"] = texture_path;
                    else if (key == "map_kd") material["diffuse_tex"] = texture_path;
                    else if (key == "map_ks") material["specular_tex"] = texture_path;
                    else if (key == "map_ke") material["emissive_tex"] = texture_path;
                    else if (key == "map_ns") material["shininess_tex"] = texture_path;
                    else if (key == "map_d") material["opacity_tex"] = texture_path;
                    else if (key == "map_bump" || key == "bump") material["bump_tex"] = texture_path;
                    else if (key == "disp") material["displacement_tex"] = texture_path;
                    else if (key == "decal") material["decal_tex"] = texture_path;
                    else if (key == "refl") material["reflection_tex"] = texture_path;
                    else if (key == "norm") material["normal_map_tex"] = texture_path;

                    // 覆盖或设置纹理选项
                    material["texture_options"] = options;
                }
            } catch (const std::exception& e) {
                zeno::log_error("Warning: Parsing error on line {}: {} - {}", line_number, line, e.what());
                continue;
            }
        }
    }

    return materials;
}

struct ReadObjPrim : INode {
    virtual void apply() override {
        auto path = get_input2<std::string>("path");
        std::string native_path = std::filesystem::u8path(path).string();
        std::ifstream file(native_path, std::ios::binary);
        auto binary = std::vector<char>((std::istreambuf_iterator<char>(file)),
                              std::istreambuf_iterator<char>());
        // auto prim = parse_obj(std::move(binary));
        auto prim = std::shared_ptr<PrimitiveObject>(parse_obj(binary.data(), binary.size()));
        if (get_param<bool>("triangulate")) {
            primTriangulate(prim.get());
        }
        set_output("prim", std::move(prim));

        auto mtl_json = std::make_shared<JsonObject>();
        auto mlt_path_str = path.substr(0, path.size() - 4) + ".mtl";
        auto mtl_path = fs::u8path(mlt_path_str);
        if (fs::exists(mtl_path)) {
            std::string native_path = mtl_path.string();
            auto content = zeno::file_get_content(native_path);
            auto hint_dir = std::filesystem::u8path(path).parent_path().u8string();
            mtl_json->json = convert_mtl_to_json(content, hint_dir);
        }

        set_output("mtl_json", mtl_json);
        std::string mtl_python = R"(
json_data = '''
mtl_json
'''

import json
mats = json.loads(json_data)

import zeno
mainG = zeno.graph("main")

index = 0
for name, mat in mats.items():
    forknode = mainG.forkAndCreate("DefaultModelShader", "Mat_{}".format(name))
    forknode.mtlid = mat['name']

    forknode.ambient_tex = mat['ambient_tex']
    forknode.ambient_value = mat['ambient_value']
    forknode.diffuse_tex = mat['diffuse_tex']
    forknode.diffuse_value = mat['diffuse_value']
    forknode.emissive_tex = mat['emissive_tex']
    forknode.emissive_value = mat['emissive_value']
    forknode.shininess_tex = mat['shininess_tex']
    forknode.shininess_value = mat['shininess_value']
    forknode.specular_tex = mat['specular_tex']
    forknode.specular_value = mat['specular_value']
    forknode.opacity_tex = mat['opacity_tex']
    forknode.opacity_value = mat['opacity_value']
    forknode.opacity_mode = 'R' if mat['opacity_tex'] != mat['diffuse_tex'] else 'A'
    forknode.bump_tex = mat['bump_tex']
    forknode.normal_map_tex = mat['normal_map_tex']
    forknode.pos = (index * 1000, 0)
    forknode.view = True
    index += 1
)";
        mtl_python = replace_all(mtl_python, "mtl_json", mtl_json->json.dump());
        set_output2("mtl_python", mtl_python);
    }
};

ZENDEFNODE(ReadObjPrim,
        { /* inputs: */ {
        {"readpath", "path"},
        }, /* outputs: */ {
        {"primitive", "prim"},
        {"mtl_json"},
        {"mtl_python"},
        }, /* params: */ {
        {"bool", "triangulate", "1"},
        }, /* category: */ {
        "primitive",
        }});

struct MustReadObjPrim : INode {
    virtual void apply() override {
        auto path = get_input2<std::string>("path");
        auto binary = file_get_binary<std::vector<char>>(path);
        if (binary.empty()) {
            auto s = zeno::format("can not find {}", path);
            throw zeno::makeError(s);
        }
        auto prim = std::shared_ptr<PrimitiveObject>(parse_obj(binary.data(), binary.size()));
        if (get_param<bool>("triangulate")) {
            primTriangulate(prim.get());
        }
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(MustReadObjPrim,
        { /* inputs: */ {
        {"readpath", "path"},
        }, /* outputs: */ {
        {"primitive", "prim"},
        }, /* params: */ {
        {"bool", "triangulate", "1"},
        }, /* category: */ {
        "primitive",
        }});
}

PrimitiveObject* primParsedFrom(const char *binData, std::size_t binSize) {
    return parse_obj(binData, binSize);
}

}