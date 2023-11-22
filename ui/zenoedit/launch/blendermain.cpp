#include <QtWidgets>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <zenomodel/include/api.h>
#include <zenomodel/include/igraphsmodel.h>
#include <zenomodel/include/graphsmanagment.h>
#include "json.hpp"

using Path = std::filesystem::path;
using Json = nlohmann::json;

struct MatChannelInfo {
    float uv_scale_x = 1.0f, uv_scale_y = 1.0f, uv_scale_z = 1.0f;
    int tex_id = -1;
    std::string separate = "";
    std::string texture_path = "";
};


int blender_main(const QCoreApplication& app);
int blender_main(const QCoreApplication& app)
{
    QCommandLineParser cmdParser;
    cmdParser.addHelpOption();
    cmdParser.addOptions({
        {"blender", "blender", ""},
        {"script", "script", "script file"},
        {"templateZsg", "templateZsg", "template zsg file"},
        {"blenderFile", "blenderFile", "blender scene file"},
        {"outputPath", "outputPath", "output zsg path"}
    });

    cmdParser.process(app);
    if (!cmdParser.isSet("blenderFile")) {
        return -1;
    }
    else if (!cmdParser.isSet("outputPath")) {
        return -1;
    }
    else if (!cmdParser.isSet("script")) {
        return -1;
    }
    else if (!cmdParser.isSet("templateZsg")) {
        return -1;
    }

    QString blenderFile = cmdParser.value("blenderFile");
    QString outputPath = cmdParser.value("outputPath");
    QString script = cmdParser.value("script");
    QString templateZsg = cmdParser.value("templateZsg");

    QStringList args = {
        script,
        "-i", blenderFile,
        "-o", outputPath
    };

    QString cmd = QString("python \"%1\" -i \"%2\" -o \"%3\"")
        .arg(script)
        .arg(blenderFile)
        .arg(outputPath);
    int ret = QProcess::execute(cmd);
    if (ret != 0) {
        std::cout << "python script execution failed" << std::endl;
    }

    auto& graphsmgr = GraphsManagment::instance();
    graphsmgr.openZsgFile(templateZsg);

    ZENO_HANDLE mGraph = Zeno_GetGraph("main");
    ZENO_HANDLE hGraph = Zeno_CreateGraph("BlenderParsed");

    //ZENO_HANDLE evalBlenderNode = index().internalId();
    std::pair<float, float> evalBlenderNodePos = { 0, 0 };
    //Zeno_GetPos(hGraph, evalBlenderNode, evalBlenderNodePos);

    // SubGraph Output
    int out_view_count = 0;
    ZENO_HANDLE suboutput_Node = Zeno_AddNode(hGraph, "SubOutput");
    Zeno_SetPos(hGraph, suboutput_Node, evalBlenderNodePos);
    ZENO_HANDLE makelist_Node = Zeno_AddNode(hGraph, "MakeList");
    Zeno_SetPos(hGraph, makelist_Node, { evalBlenderNodePos.first - 500.0f, evalBlenderNodePos.second });
    Zeno_AddLink(hGraph, makelist_Node, "list", suboutput_Node, "port");

    Path dir(outputPath.toStdString());
    Path json_file("info.json");
    Path zsg_file("parsed.zsg");
    Path out_json = dir / json_file;
    Path out_zsg = dir / zsg_file;

    if (!exists(out_json)) {
        std::cout << "error: the info.json that script output doesn't exists\n";
        return -1;
    }
    std::ifstream f(out_json.string());
    std::string str;
    if (f) {
        std::ostringstream ss;
        ss << f.rdbuf(); // reading data
        str = ss.str();
    }
    Json parseData = Json::parse(str);

    int abc_count = 0;
    // Eval Mesh
    for (auto& [key, val] : parseData.items())
    {
        Path file = Path(key + ".abc");
        Path out_abc = dir / Path("abc") / file;
        std::cout << "abc file: " << out_abc.string() << " exists: " << exists(out_abc) << "\n";
        if (exists(out_abc))
        {
            std::pair<float, float> NodePos1 = { evalBlenderNodePos.first + 600.0f, evalBlenderNodePos.second + 300.0f * abc_count };
            std::pair<float, float> NodePos2 = { evalBlenderNodePos.first + 1100.0f, evalBlenderNodePos.second + 300.0f * abc_count };
            std::pair<float, float> NodePos3 = { evalBlenderNodePos.first + 1600.0f, evalBlenderNodePos.second + 300.0f * abc_count };

            // Create nodes
            ZENO_HANDLE read_alembic_Node = Zeno_AddNode(hGraph, "ReadAlembic");
            Zeno_SetPos(hGraph, read_alembic_Node, NodePos1);

            ZENO_HANDLE alembic_prim_Node = Zeno_AddNode(hGraph, "AllAlembicPrim");
            Zeno_SetPos(hGraph, alembic_prim_Node, NodePos2);

            ZENO_HANDLE bind_material_Node = Zeno_AddNode(hGraph, "BindMaterial");
            Zeno_SetPos(hGraph, bind_material_Node, NodePos3);

            // Set node inputs
            Zeno_SetInputDefl(hGraph, read_alembic_Node, "path", out_abc.string());
            Zeno_SetInputDefl(hGraph, alembic_prim_Node, "use_xform", true);
            Zeno_SetInputDefl(hGraph, alembic_prim_Node, "triangulate", true);
            Zeno_SetInputDefl(hGraph, bind_material_Node, "mtlid", key);

            // Link nodes
            Zeno_AddLink(hGraph, read_alembic_Node, "abctree", alembic_prim_Node, "abctree");
            Zeno_AddLink(hGraph, alembic_prim_Node, "prim", bind_material_Node, "object");

            // Set node view
            //Zeno_SetView(hGraph, bind_material_Node, true);

            Zeno_AddLink(hGraph, bind_material_Node, "object", makelist_Node, "obj" + std::to_string(out_view_count));
            out_view_count++;

            abc_count++;
        }
        else {
            std::cout << "warn: " << out_abc.string() << " doesn't exists, skip\n";
        }
    }

    int mat_count = 0;
    // Eval Material
    for (auto& [key, val] : parseData.items()) {

        std::vector<std::string> mat_texs{};
        std::map<std::string, MatChannelInfo> mat_infos{};

        if (val.find("_textures") != val.end())
        {
            for (auto& [_mat_channel, val_] : val["_textures"].items())
            {
                MatChannelInfo channel_info{};

                // only have one element, and we are bound to get into this cycle
                for (auto& [_tex_path, val__] : val_.items()) {
                    auto channel_name = val__["channel_name"];
                    auto mapping_scale = val__["mapping_scale"];
                    auto separate_name = val__["separate_name"];
                    auto scale_value = mapping_scale.get<std::vector<float>>();
                    channel_info.uv_scale_x = scale_value[0];
                    channel_info.uv_scale_y = scale_value[1];
                    channel_info.uv_scale_z = scale_value[2];

                    auto separate_value = separate_name.get<std::string>();
                    channel_info.separate = separate_value;
                    channel_info.texture_path = _tex_path;
                    if (std::find(mat_texs.begin(), mat_texs.end(), _tex_path) == mat_texs.end()) {
                        channel_info.tex_id = mat_texs.size();
                        // If the element is not found, insert it
                        mat_texs.emplace_back(_tex_path);
                    }
                    else {
                        int index = (int)std::distance(mat_texs.begin(), std::find(mat_texs.begin(), mat_texs.end(), _tex_path));
                        channel_info.tex_id = index;
                    }
                    break;
                }
                mat_infos[_mat_channel] = channel_info;
            }
        }

        std::pair<float, float> NodePos2 = { evalBlenderNodePos.first + 3000.0f * mat_count + 800.0f, evalBlenderNodePos.second - 2000.0f };
        ZENO_HANDLE shader_finalize_Node = Zeno_AddNode(hGraph, "ShaderFinalize");
        Zeno_SetPos(hGraph, shader_finalize_Node, NodePos2);
        Zeno_SetInputDefl(hGraph, shader_finalize_Node, "mtlid", key);
        //Zeno_SetView(hGraph, shader_finalize_Node, true);

        Zeno_AddLink(hGraph, shader_finalize_Node, "mtl", makelist_Node, "obj" + std::to_string(out_view_count));
        out_view_count++;

        if (val.find("base_color") != val.end()) {
            auto base_color = val.at("base_color").at("value").get<std::vector<float>>(); // vec4
            auto subsurface = val.at("subsurface").at("value").get<float>();
            auto subsurface_radius = val.at("subsurface_radius").at("value").get<std::vector<float>>(); // vec3
            auto subsurface_color = val.at("subsurface_color").at("value").get<std::vector<float>>(); // vec4
            auto subsurface_ior = val.at("subsurface_ior").at("value").get<float>();
            auto subsurface_anisotropy = val.at("subsurface_anisotropy").at("value").get<float>();
            auto metallic = val.at("metallic").at("value").get<float>();
            auto specular = val.at("specular").at("value").get<float>();
            auto specular_tint = val.at("specular_tint").at("value").get<float>();
            auto roughness = val.at("roughness").at("value").get<float>();
            auto anisotropic = val.at("anisotropic").at("value").get<float>();
            auto anisotropic_rotation = val.at("anisotropic_rotation").at("value").get<float>();
            auto sheen = val.at("sheen").at("value").get<float>();
            auto sheen_tint = val.at("sheen_tint").at("value").get<float>();
            auto sheen_roughness = val.at("sheen_roughness").at("value").get<float>();
            auto clearcoat = val.at("clearcoat").at("value").get<float>();
            auto clearcoat_roughness = val.at("clearcoat_roughness").at("value").get<float>();
            auto ior = val.at("ior").at("value").get<float>();
            auto transmission = val.at("transmission").at("value").get<float>();
            auto emission = val.at("emission").at("value").get<std::vector<float>>(); // vec4
            auto emission_strength = val.at("emission_strength").at("value").get<float>();
            auto alpha = val.at("alpha").at("value").get<float>();
            auto normal = val.at("normal").at("value").get<std::vector<float>>(); //vec3
            auto clearcoat_normal = val.at("clearcoat_normal").at("value").get<std::vector<float>>(); //vec3
            auto tangent = val.at("tangent").at("value").get<std::vector<float>>(); //vec3

            Zeno_SetInputDefl(hGraph, shader_finalize_Node, "basecolor", zeno::vec3f(base_color[0], base_color[1], base_color[2]));
            Zeno_SetInputDefl(hGraph, shader_finalize_Node, "metallic", metallic);
            Zeno_SetInputDefl(hGraph, shader_finalize_Node, "roughness", roughness);
            Zeno_SetInputDefl(hGraph, shader_finalize_Node, "specular", specular);
            Zeno_SetInputDefl(hGraph, shader_finalize_Node, "subsurface", subsurface);
            Zeno_SetInputDefl(hGraph, shader_finalize_Node, "sssParam", zeno::vec3f(subsurface_radius[0], subsurface_radius[1], subsurface_radius[2]));
            Zeno_SetInputDefl(hGraph, shader_finalize_Node, "sssColor", zeno::vec3f(subsurface_color[0], subsurface_color[1], subsurface_color[2]));
            Zeno_SetInputDefl(hGraph, shader_finalize_Node, "specularTint", specular_tint);
            Zeno_SetInputDefl(hGraph, shader_finalize_Node, "anisotropic", anisotropic);
            Zeno_SetInputDefl(hGraph, shader_finalize_Node, "anisoRotation", anisotropic_rotation);
            Zeno_SetInputDefl(hGraph, shader_finalize_Node, "sheen", sheen);
            Zeno_SetInputDefl(hGraph, shader_finalize_Node, "sheenTint", sheen_tint);
            Zeno_SetInputDefl(hGraph, shader_finalize_Node, "clearcoat", clearcoat);
            Zeno_SetInputDefl(hGraph, shader_finalize_Node, "clearcoatRoughness", clearcoat_roughness);
            Zeno_SetInputDefl(hGraph, shader_finalize_Node, "specTrans", transmission);
            Zeno_SetInputDefl(hGraph, shader_finalize_Node, "ior", ior);
            Zeno_SetInputDefl(hGraph, shader_finalize_Node, "emission", zeno::vec3f(emission[0], emission[1], emission[2]));
            Zeno_SetInputDefl(hGraph, shader_finalize_Node, "emissionIntensity", emission_strength);
        }

        if (mat_infos.size())
        {
            std::pair<float, float> NodePos0 = { evalBlenderNodePos.first + 3000.0f * mat_count, evalBlenderNodePos.second };
            std::pair<float, float> NodePos1 = { evalBlenderNodePos.first + 3000.0f * mat_count, evalBlenderNodePos.second + 1000.0f };

            ZENO_HANDLE shader_attr_Node = Zeno_AddNode(hGraph, "ShaderInputAttr");
            Zeno_SetPos(hGraph, shader_attr_Node, NodePos0);
            Zeno_SetInputDefl(hGraph, shader_attr_Node, "attr", std::string("uv"));

            int shader_texture_count = 0;
            for (auto& [key, value] : mat_infos) {
                std::pair<float, float> NodePos_1 = { NodePos0.first + 600.0f, NodePos0.second + 500.0f * shader_texture_count };
                std::pair<float, float> NodePos_2 = { NodePos_1.first + 600.0f, NodePos_1.second };
                std::pair<float, float> NodePos_3 = { NodePos_2.first + 600.0f, NodePos_2.second };

                ZENO_HANDLE uv_transform_Node = Zeno_AddNode(hGraph, "FBXUVTransform");
                Zeno_SetPos(hGraph, uv_transform_Node, NodePos_1);
                ZENO_HANDLE shader_texture_Node = Zeno_AddNode(hGraph, "ShaderTexture2D");
                Zeno_SetPos(hGraph, shader_texture_Node, NodePos_2);


                Zeno_AddLink(hGraph, shader_attr_Node, "out", uv_transform_Node, "uvattr");
                Zeno_AddLink(hGraph, uv_transform_Node, "uvw", shader_texture_Node, "coord");

                Zeno_SetInputDefl(hGraph, shader_texture_Node, "texId", value.tex_id);

                Zeno_SetInputDefl(hGraph, uv_transform_Node, "uvtransform", zeno::vec4f(value.uv_scale_x, value.uv_scale_y, 0.0f, 0.0f));
#define PARAM_TYPE_CHECK(to_channel_name) \
if( \
    #to_channel_name == "metallic" || \
    #to_channel_name == "roughness" || \
    #to_channel_name == "specular" || \
    #to_channel_name == "subsurface" || \
    #to_channel_name == "specularTint" || \
    #to_channel_name == "anisotropic" || \
    #to_channel_name == "anisoRotation" || \
    #to_channel_name == "sheen" || \
    #to_channel_name == "sheenTint" || \
    #to_channel_name == "clearcoat" || \
    #to_channel_name == "clearcoatRoughness" || \
    #to_channel_name == "specTrans" || \
    #to_channel_name == "ior" || \
    #to_channel_name == "emissionIntensity") \
{ \
   ZENO_HANDLE extrac_vec_Node = Zeno_AddNode(hGraph, "ShaderExtractVec"); \
   Zeno_SetPos(hGraph, extrac_vec_Node, NodePos_3); \
   Zeno_AddLink(hGraph, shader_texture_Node, "out", extrac_vec_Node, "vec"); \
   Zeno_AddLink(hGraph, extrac_vec_Node, "x", shader_finalize_Node, #to_channel_name); \
}else{                                    \
   Zeno_AddLink(hGraph, shader_texture_Node, "out", shader_finalize_Node, #to_channel_name);  \
}

#define PARAM_SEPARATE(to_channel_name)   \
if(value.separate != ""){ \
    ZENO_HANDLE extrac_vec_Node = Zeno_AddNode(hGraph, "ShaderExtractVec"); \
    Zeno_SetPos(hGraph, extrac_vec_Node, NodePos_3); \
    Zeno_AddLink(hGraph, shader_texture_Node, "out", extrac_vec_Node, "vec"); \
    if(value.separate == "Red"){ \
        Zeno_AddLink(hGraph, extrac_vec_Node, "x", shader_finalize_Node, #to_channel_name); \
    }else if(value.separate == "Green"){ \
        Zeno_AddLink(hGraph, extrac_vec_Node, "y", shader_finalize_Node, #to_channel_name); \
    }else if(value.separate == "Blue"){ \
        Zeno_AddLink(hGraph, extrac_vec_Node, "z", shader_finalize_Node, #to_channel_name); \
    } \
}else{ \
    PARAM_TYPE_CHECK(to_channel_name) \
} \

                if (key == "Base Color") {
                    PARAM_SEPARATE(basecolor)
                }
                else if (key == "Subsurface") {
                    PARAM_SEPARATE(subsurface)
                }
                else if (key == "Subsurface Radius") {
                    PARAM_SEPARATE(sssParam)
                }
                else if (key == "Subsurface Color") {
                    PARAM_SEPARATE(sssColor)
                }
                else if (key == "Metallic") {
                    PARAM_SEPARATE(metallic)
                }
                else if (key == "Specular") {
                    PARAM_SEPARATE(specular)
                }
                else if (key == "Specular Tint") {
                    PARAM_SEPARATE(specularTint)
                }
                else if (key == "Roughness") {
                    PARAM_SEPARATE(roughness)
                }
                else if (key == "Anisotropic") {
                    PARAM_SEPARATE(anisotropic)
                }
                else if (key == "Anisotropic Rotation") {
                    PARAM_SEPARATE(anisoRotation)
                }
                else if (key == "Sheen") {
                    PARAM_SEPARATE(sheen)
                }
                else if (key == "Sheen Tint") {
                    PARAM_SEPARATE(sheenTint)
                }
                else if (key == "Clearcoat") {
                    PARAM_SEPARATE(clearcoat)
                }
                else if (key == "Clearcoat Roughness") {
                    PARAM_SEPARATE(clearcoatRoughness)
                }
                else if (key == "IOR") {
                    PARAM_SEPARATE(ior)
                }
                else if (key == "Transmission") {
                    PARAM_SEPARATE(specTrans)
                }
                else if (key == "Emission") {
                    PARAM_SEPARATE(emission)
                }
                else if (key == "Emission Strength") {
                    PARAM_SEPARATE(emissionIntensity)
                }

                if (key == "Normal") {
                    ZENO_HANDLE normal_texture_Node = Zeno_AddNode(hGraph, "NormalTexture");
                    Zeno_SetPos(hGraph, normal_texture_Node, NodePos_2);
                    Zeno_DeleteNode(hGraph, shader_texture_Node);

                    Zeno_SetInputDefl(hGraph, normal_texture_Node, "texId", value.tex_id);
                    Zeno_AddLink(hGraph, uv_transform_Node, "uvw", normal_texture_Node, "uv");
                    Zeno_AddLink(hGraph, normal_texture_Node, "normal", shader_finalize_Node, "normal");
                }
                shader_texture_count++;
            }

            ZENO_HANDLE make_list_Node = Zeno_AddNode(hGraph, "MakeList");
            Zeno_SetPos(hGraph, make_list_Node, NodePos1);

            int texture_2d_count = 0;
            for (auto& tex_path : mat_texs) {
                std::pair<float, float> NodePos_1 = { NodePos1.first - 700.0f, NodePos1.second + 500.0f * texture_2d_count };

                ZENO_HANDLE texture_2d_Node = Zeno_AddNode(hGraph, "MakeTexture2D");
                Zeno_SetPos(hGraph, texture_2d_Node, NodePos_1);

                Zeno_AddLink(hGraph, texture_2d_Node, "tex", make_list_Node, "obj" + std::to_string(texture_2d_count));

                Zeno_SetInputDefl(hGraph, texture_2d_Node, "path", tex_path);

                texture_2d_count++;
            }

            Zeno_AddLink(hGraph, make_list_Node, "list", shader_finalize_Node, "tex2dList");
        }

        mat_count++;
    }

    // Create Parsed-SubNode in main
    ZENO_HANDLE parsed_blender_Node = Zeno_AddNode(mGraph, "BlenderParsed");
    Zeno_SetPos(mGraph, parsed_blender_Node, { evalBlenderNodePos.first + 500.0f, evalBlenderNodePos.second });
    Zeno_SetView(mGraph, parsed_blender_Node, true);

    // Save zsg file
    IGraphsModel* pModel = GraphsManagment::instance().currentModel();
    GraphsManagment& gman = GraphsManagment::instance();
    APP_SETTINGS settings;
    TIMELINE_INFO info;
    settings.timeline = info;
    std::cout << "save zsg file: " << out_zsg.string() << "\n";
    gman.saveFile(QString::fromStdString(out_zsg.string()), settings);

    return 0;
}