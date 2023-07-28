#include "readfbxprim.h"
#include "util/log.h"
#include <zenomodel/include/api.h>
#include <zenomodel/include/igraphsmodel.h>
#include "zenoapplication.h"
#include <zenomodel/include/graphsmanagment.h>

#include <viewport/zenovis.h>
#include <zeno/core/Session.h>
#include <zenovis/ObjectsManager.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/types/UserData.h>
#include <zeno/extra/TempNode.h>
#include <zeno/types/StringObject.h>

#include <filesystem>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "json.hpp"

using Path = std::filesystem::path;
using Json = nlohmann::json;

int my_sqrt(int x) {
    // Base case
    if (x == 0 || x == 1) {
        return x;
    }

    // Initialize variables
    int y = x;
    int z = 1;

    // Apply Babylonian method until convergence
    while (y > z) {
        y = (y + z) / 2;
        z = x / y;
    }

    // Return the floor of the final square root
    return y;
}

ReadFBXPrim::ReadFBXPrim(const NodeUtilParam& params, QGraphicsItem* parent)
    : ZenoNode(params, parent)
{
    GenMode = -1;
}

ReadFBXPrim::~ReadFBXPrim()
{

}

ZGraphicsLayout* ReadFBXPrim::initCustomParamWidgets()
{
    ZGraphicsLayout* pHLayout = new ZGraphicsLayout(true);

    ZGraphicsLayout* pHLayoutNode = new ZGraphicsLayout(true);
    ZGraphicsLayout* pHLayoutPart = new ZGraphicsLayout(true);

    ZSimpleTextItem* pNodeItem = new ZSimpleTextItem("node");
    pNodeItem->setBrush(m_renderParams.socketClr.color());
    pNodeItem->setFont(m_renderParams.socketFont);
    pNodeItem->updateBoundingRect();
    pHLayoutNode->addItem(pNodeItem);

    ZSimpleTextItem* pPartItem = new ZSimpleTextItem("part");
    pPartItem->setBrush(m_renderParams.socketClr.color());
    pPartItem->setFont(m_renderParams.socketFont);
    pPartItem->updateBoundingRect();
    pHLayoutPart->addItem(pPartItem);

    ZenoParamPushButton* pNodeBtn = new ZenoParamPushButton("Generate", -1, QSizePolicy::Expanding);
    pHLayoutNode->addItem(pNodeBtn);
    connect(pNodeBtn, SIGNAL(clicked()), this, SLOT(onNodeClicked()));

    ZenoParamPushButton* pPartBtn = new ZenoParamPushButton("Generate", -1, QSizePolicy::Expanding);
    pHLayoutPart->addItem(pPartBtn);
    connect(pPartBtn, SIGNAL(clicked()), this, SLOT(onPartClicked()));

    _param_ctrl paramNode;
    _param_ctrl paramPart;
    paramNode.param_name = pNodeItem;
    paramNode.param_control = pNodeBtn;
    paramNode.ctrl_layout = pHLayoutNode;

    paramPart.param_name = pPartItem;
    paramPart.param_control = pPartBtn;
    paramPart.ctrl_layout = pHLayoutPart;

    addParam(paramNode);
    addParam(paramPart);

    pHLayout->addLayout(pHLayoutNode);
    pHLayout->addLayout(pHLayoutPart);

    return pHLayout;
}
void ReadFBXPrim::GenerateFBX() {
    std::cout << "Generate FBX, Part " << GenMode << "\n";
    ZENO_HANDLE hGraph = Zeno_GetGraph("main");

    // Get ReadFBXPrim ident
    ZENO_HANDLE fbxNode = index().internalId();

    // Get the position of current ReadFBXPrim node
    std::pair<float, float> fbxNodePos;
    Zeno_GetPos(hGraph, fbxNode, fbxNodePos);

    // Get FBX Path
    ZVARIANT path; std::string type;
    Zeno_GetInputDefl(hGraph, fbxNode, "path", path, type);

    // Get FBX HintPath
    ZVARIANT hintPath;
    Zeno_GetInputDefl(hGraph, fbxNode, "hintPath", hintPath, type);

    std::string get_path = std::get<std::string>(path);
    std::string get_hintPath = std::get<std::string>(hintPath);

    auto path_ = std::make_shared<zeno::StringObject>(); path_->set(get_path);
    auto hintPath_ = std::make_shared<zeno::StringObject>(); hintPath_->set(get_hintPath);

    auto outs = zeno::TempNodeSimpleCaller("ReadFBXPrim")
                    .set("path", path_)
                    .set("hintPath", hintPath_)
                    .set2<bool>("generate", true)
                    .set2<float>("offset", 0.0)
                    .set2<std::string>("udim:", "DISABLE")
                    .set2<bool>("primitive:", false)
                    .set2<bool>("printTree:", false)
                    .set2<bool>("triangulate:", true)
                    .set2<bool>("indepData:", false)
                    .call();
    zeno::log_info("ReadFBXPrim Caller End");

    // Create nodes
    auto fbxObj = outs.get("prim");

    if(fbxObj) {
        auto matNum = fbxObj->userData().getLiterial<int>("matNum");
        auto fbxName = fbxObj->userData().getLiterial<std::string>("fbxName");

        ZENO_HANDLE fbxPartGraph = Zeno_GetGraph("FBXPart");
        ZASSERT_EXIT(fbxPartGraph);

        int check = my_sqrt(matNum);
        int my_i = 0;
        float add_x_pos = 0.0f;
        float add_y_pos = 0.0f;

        for (int i = 0; i < matNum; i++) {
            auto matName = fbxObj->userData().getLiterial<std::string>(std::to_string(i));

            std::cout<<"total:"<<matNum<<", current:"<<i<<"\n";

            // Pos
            add_y_pos = my_i * 300.0f;
            my_i++;
            std::pair<float, float> firstNodePos = {fbxNodePos.first + 600.0f + add_x_pos, fbxNodePos.second + add_y_pos};
            if(my_i % check == 0){
                add_x_pos += 1500.0f;
                my_i = 0;
            }

            // Name
            std::string concatname = matName;
            if (matName.find(':') != std::string::npos) {
                std::replace(concatname.begin(), concatname.end(), ':', '_');
            }
            std::string fbxPartGraphName = fbxName+"_"+concatname;

            if(GenMode == 0){

                ZENO_HANDLE dictNode = Zeno_AddNode(hGraph, "DictGetItem");
                Zeno_SetPos(hGraph,dictNode, firstNodePos);
                Zeno_SetInputDefl(hGraph, dictNode, "key", matName);
                Zeno_AddLink(hGraph, fbxNode, "mats", dictNode, "dict");

                ZENO_HANDLE forkedSubg = 0;
                ZENO_HANDLE forkedNode = 0;

                ZENO_ERROR ret = Zeno_ForkGraph(hGraph, "FBXPart", forkedSubg, forkedNode);
                ZASSERT_EXIT(!ret);ZASSERT_EXIT(forkedSubg);ZASSERT_EXIT(forkedNode);

                Zeno_RenameGraph(forkedSubg, fbxPartGraphName);

                std::pair<float, float> fbxPartPos = {firstNodePos.first + 500.0f, firstNodePos.second};

                Zeno_SetPos(hGraph, forkedNode, fbxPartPos);
                Zeno_AddLink(hGraph, dictNode, "object", forkedNode, "data");
                Zeno_SetView(hGraph, forkedNode, true);

            }else if(GenMode == 1){

                ZENO_HANDLE forkedSubg = 0;
                ZENO_HANDLE forkedNode = 0;
                ZENO_ERROR ret = Zeno_ForkGraph(hGraph, "FBXMatPart", forkedSubg, forkedNode);
                ZASSERT_EXIT(!ret);ZASSERT_EXIT(forkedSubg);ZASSERT_EXIT(forkedNode);
                Zeno_RenameGraph(forkedSubg, fbxPartGraphName);

                std::pair<float, float> fbxPartPos = {firstNodePos.first + 500.0f, firstNodePos.second};
                Zeno_SetPos(hGraph, forkedNode, fbxPartPos);
                Zeno_AddLink(hGraph, fbxNode, "mats", forkedNode, "mats");
                Zeno_SetInputDefl(hGraph, forkedNode, "matname", matName);
                Zeno_SetView(hGraph, forkedNode, true);
            }
        }

        if(GenMode == 1){
            ZENO_HANDLE forkedSubg = 0;
            ZENO_HANDLE forkedNode = 0;
            ZENO_ERROR ret = Zeno_ForkGraph(hGraph, "FBXMeshPart", forkedSubg, forkedNode);
            ZASSERT_EXIT(!ret);ZASSERT_EXIT(forkedSubg);ZASSERT_EXIT(forkedNode);
            auto fbxMeshPartGraphName = fbxName+"_FBXMeshPart";
            Zeno_RenameGraph(forkedSubg, fbxMeshPartGraphName);
            std::pair<float, float> fbxMeshPartPos = {fbxNodePos.first + 1000.0f, fbxNodePos.second - 600.0f};
            Zeno_SetPos(hGraph, forkedNode, fbxMeshPartPos);
            Zeno_AddLink(hGraph, fbxNode, "datas", forkedNode, "datas");
            Zeno_SetView(hGraph, forkedNode, true);
        }

    }else{
        zeno::log_error("Not found ReadFBXPrim node in objectsMan");
    }
}

void ReadFBXPrim::onNodeClicked()
{
    GenMode = 0;
    GenerateFBX();
}

void ReadFBXPrim::onPartClicked()
{
    GenMode = 1;
    GenerateFBX();
}

EvalBlenderFile::EvalBlenderFile(const NodeUtilParam& params, QGraphicsItem* parent)
    : ZenoNode(params, parent)
{

}

EvalBlenderFile::~EvalBlenderFile()
{

}

ZGraphicsLayout* EvalBlenderFile::initCustomParamWidgets()
{
    ZGraphicsLayout* pHLayout = new ZGraphicsLayout(false);

    ZGraphicsLayout* pHLayoutExec = new ZGraphicsLayout(true);
    ZGraphicsLayout* pHLayoutEval = new ZGraphicsLayout(true);

    ZSimpleTextItem* pExecItem = new ZSimpleTextItem("exec");
    pExecItem->setBrush(m_renderParams.socketClr.color());
    pExecItem->setFont(m_renderParams.socketFont);
    pExecItem->updateBoundingRect();
    pHLayoutExec->addItem(pExecItem);

    ZSimpleTextItem* pEvalItem = new ZSimpleTextItem("eval");
    pEvalItem->setBrush(m_renderParams.socketClr.color());
    pEvalItem->setFont(m_renderParams.socketFont);
    pEvalItem->updateBoundingRect();
    pHLayoutEval->addItem(pEvalItem);

    ZenoParamPushButton* pExecBtn = new ZenoParamPushButton("Generate", -1, QSizePolicy::Expanding);
    pHLayoutExec->addItem(pExecBtn);
    connect(pExecBtn, SIGNAL(clicked()), this, SLOT(onExecClicked()));

    ZenoParamPushButton* pEvalBtn = new ZenoParamPushButton("Generate", -1, QSizePolicy::Expanding);
    pHLayoutEval->addItem(pEvalBtn);
    connect(pEvalBtn, SIGNAL(clicked()), this, SLOT(onEvalClicked()));

    _param_ctrl paramNode;
    _param_ctrl paramPart;
    paramNode.param_name = pExecItem;
    paramNode.param_control = pExecBtn;
    paramNode.ctrl_layout = pHLayoutExec;

    paramPart.param_name = pEvalItem;
    paramPart.param_control = pEvalBtn;
    paramPart.ctrl_layout = pHLayoutEval;

    addParam(paramNode);
    addParam(paramPart);

    pHLayout->addLayout(pHLayoutExec);
    pHLayout->addLayout(pHLayoutEval);

    return pHLayout;
}

struct MatChannelInfo{
    float uv_scale_x = 1.0f, uv_scale_y = 1.0f, uv_scale_z = 1.0f;
    int tex_id = -1;
    std::string separate = "";
    std::string texture_path = "";
};

void EvalBlenderFile::onEvalClicked() {
    ZENO_HANDLE hGraph = Zeno_GetGraph("main");

    ZENO_HANDLE evalBlenderNode = index().internalId();
    std::pair<float, float> evalBlenderNodePos;
    Zeno_GetPos(hGraph, evalBlenderNode, evalBlenderNodePos);

    auto inputs = GetCurNodeInputs();
    std::string output_path = inputs[2];
    Path dir (output_path);
    Path file ("info.json");
    Path out_json = dir / file;
    if(! exists(out_json)){
        std::cout << "error: the info.json that script output doesn't exists\n";
        return;
    }
    std::ifstream f(out_json.string());
    std::string str;
    if(f) {
        std::ostringstream ss;
        ss << f.rdbuf(); // reading data
        str = ss.str();
    }
    Json parseData = Json::parse(str);

    int abc_count = 0;
    // Eval Mesh
    for (auto& [key, val] : parseData.items())
    {
        file = Path(key+".abc");
        Path out_abc = dir / Path("abc") / file;
        std::cout << "abc file: " << out_abc.string() << " exists: " << exists(out_abc) << "\n";
        if(exists(out_abc))
        {
            std::pair<float, float> NodePos1 = { evalBlenderNodePos.first + 600.0f, evalBlenderNodePos.second + 300.0f * abc_count};
            std::pair<float, float> NodePos2 = { evalBlenderNodePos.first + 1100.0f, evalBlenderNodePos.second + 300.0f * abc_count};
            std::pair<float, float> NodePos3 = { evalBlenderNodePos.first + 1600.0f, evalBlenderNodePos.second + 300.0f * abc_count};

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
            Zeno_SetView(hGraph, bind_material_Node, true);

            abc_count++;
        }else{
            std::cout << "warn: " << out_abc.string() << " doesn't exists, skip\n";
        }
    }

    int mat_count = 0;
    // Eval Material
    for (auto& [key, val] : parseData.items()){

        std::vector<std::string> mat_texs{};
        std::map<std::string, MatChannelInfo> mat_infos{};

        if(val.find("_textures") != val.end())
        {
            for(auto& [_mat_channel, val_] : val["_textures"].items())
            {
                MatChannelInfo channel_info{};

                // only have one element, and we are bound to get into this cycle
                for(auto& [_tex_path, val__] : val_.items()){
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
                    }else{
                        int index = (int)std::distance(mat_texs.begin(), std::find(mat_texs.begin(), mat_texs.end(), _tex_path));
                        channel_info.tex_id = index;
                    }
                    break;
                }
                mat_infos[_mat_channel] = channel_info;
            }
        }

        std::pair<float, float> NodePos2 = { evalBlenderNodePos.first + 3000.0f * mat_count + 800.0f, evalBlenderNodePos.second - 2000.0f};
        ZENO_HANDLE shader_finalize_Node = Zeno_AddNode(hGraph, "ShaderFinalize");
        Zeno_SetPos(hGraph, shader_finalize_Node, NodePos2);
        Zeno_SetInputDefl(hGraph, shader_finalize_Node, "mtlid", key);
        Zeno_SetView(hGraph, shader_finalize_Node, true);

        if(val.find("base_color") != val.end()){
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

          Zeno_SetInputDefl(hGraph, shader_finalize_Node, "basecolor", zeno::vec3f(base_color[0],base_color[1],base_color[2]));
          Zeno_SetInputDefl(hGraph, shader_finalize_Node, "metallic", metallic);
          Zeno_SetInputDefl(hGraph, shader_finalize_Node, "roughness", roughness);
          Zeno_SetInputDefl(hGraph, shader_finalize_Node, "specular", specular);
          Zeno_SetInputDefl(hGraph, shader_finalize_Node, "subsurface", subsurface);
          Zeno_SetInputDefl(hGraph, shader_finalize_Node, "sssParam", zeno::vec3f(subsurface_radius[0],subsurface_radius[1],subsurface_radius[2]));
          Zeno_SetInputDefl(hGraph, shader_finalize_Node, "sssColor", zeno::vec3f(subsurface_color[0],subsurface_color[1],subsurface_color[2]));
          Zeno_SetInputDefl(hGraph, shader_finalize_Node, "specularTint", specular_tint);
          Zeno_SetInputDefl(hGraph, shader_finalize_Node, "anisotropic", anisotropic);
          Zeno_SetInputDefl(hGraph, shader_finalize_Node, "anisoRotation", anisotropic_rotation);
          Zeno_SetInputDefl(hGraph, shader_finalize_Node, "sheen", sheen);
          Zeno_SetInputDefl(hGraph, shader_finalize_Node, "sheenTint", sheen_tint);
          Zeno_SetInputDefl(hGraph, shader_finalize_Node, "clearcoat", clearcoat);
          Zeno_SetInputDefl(hGraph, shader_finalize_Node, "clearcoatRoughness", clearcoat_roughness);
          Zeno_SetInputDefl(hGraph, shader_finalize_Node, "specTrans", transmission);
          Zeno_SetInputDefl(hGraph, shader_finalize_Node, "ior", ior);
          Zeno_SetInputDefl(hGraph, shader_finalize_Node, "emission", zeno::vec3f(emission[0],emission[1],emission[2]));
          Zeno_SetInputDefl(hGraph, shader_finalize_Node, "emissionIntensity", emission_strength);
        }

        if(mat_infos.size())
        {
            std::pair<float, float> NodePos0 = { evalBlenderNodePos.first + 3000.0f * mat_count, evalBlenderNodePos.second};
            std::pair<float, float> NodePos1 = { evalBlenderNodePos.first + 3000.0f * mat_count, evalBlenderNodePos.second + 1000.0f};

            ZENO_HANDLE shader_attr_Node = Zeno_AddNode(hGraph, "ShaderInputAttr");
            Zeno_SetPos(hGraph, shader_attr_Node, NodePos0);
            Zeno_SetInputDefl(hGraph, shader_attr_Node, "attr", "uv");

            int shader_texture_count = 0;
            for(auto &[key, value] : mat_infos) {
                std::pair<float, float> NodePos_1 = { NodePos0.first + 600.0f, NodePos0.second + 500.0f * shader_texture_count};
                std::pair<float, float> NodePos_2 = { NodePos_1.first + 600.0f, NodePos_1.second};

                ZENO_HANDLE uv_transform_Node = Zeno_AddNode(hGraph, "FBXUVTransform");
                Zeno_SetPos(hGraph, uv_transform_Node, NodePos_1);
                ZENO_HANDLE shader_texture_Node = Zeno_AddNode(hGraph, "ShaderTexture2D");
                Zeno_SetPos(hGraph, shader_texture_Node, NodePos_2);

                Zeno_AddLink(hGraph, shader_attr_Node, "out", uv_transform_Node, "uvattr");
                Zeno_AddLink(hGraph, uv_transform_Node, "uvw", shader_texture_Node, "coord");

                Zeno_SetInputDefl(hGraph, shader_texture_Node, "texId", value.tex_id);
//                Zeno_SetInputDefl(hGraph, uv_transform_Node, "uvtransform", zeno::vec4f(value.uv_scale_x, value.uv_scale_y, 0, 0));

                if(key == "Base Color")
                    Zeno_AddLink(hGraph, shader_texture_Node, "out", shader_finalize_Node, "basecolor");
                else if(key == "Subsurface")
                    Zeno_AddLink(hGraph, shader_texture_Node, "out", shader_finalize_Node, "subsurface");
                else if(key == "Subsurface Radius")
                    Zeno_AddLink(hGraph, shader_texture_Node, "out", shader_finalize_Node, "sssParam");
                else if(key == "Subsurface Color")
                    Zeno_AddLink(hGraph, shader_texture_Node, "out", shader_finalize_Node, "sssColor");
                else if(key == "Metallic")
                    Zeno_AddLink(hGraph, shader_texture_Node, "out", shader_finalize_Node, "metallic");
                else if(key == "Specular")
                    Zeno_AddLink(hGraph, shader_texture_Node, "out", shader_finalize_Node, "specular");
                else if(key == "Specular Tint")
                    Zeno_AddLink(hGraph, shader_texture_Node, "out", shader_finalize_Node, "specularTint");
                else if(key == "Roughness")
                    Zeno_AddLink(hGraph, shader_texture_Node, "out", shader_finalize_Node, "roughness");
                else if(key == "Anisotropic")
                    Zeno_AddLink(hGraph, shader_texture_Node, "out", shader_finalize_Node, "anisotropic");
                else if(key == "Anisotropic Rotation")
                    Zeno_AddLink(hGraph, shader_texture_Node, "out", shader_finalize_Node, "anisoRotation");
                else if(key == "Sheen")
                    Zeno_AddLink(hGraph, shader_texture_Node, "out", shader_finalize_Node, "sheen");
                else if(key == "Sheen Tint")
                    Zeno_AddLink(hGraph, shader_texture_Node, "out", shader_finalize_Node, "sheenTint");
                else if(key == "Clearcoat")
                    Zeno_AddLink(hGraph, shader_texture_Node, "out", shader_finalize_Node, "clearcoat");
                else if(key == "Clearcoat Roughness")
                    Zeno_AddLink(hGraph, shader_texture_Node, "out", shader_finalize_Node, "clearcoatRoughness");
                else if(key == "IOR")
                    Zeno_AddLink(hGraph, shader_texture_Node, "out", shader_finalize_Node, "ior");
                else if(key == "Transmission")
                    Zeno_AddLink(hGraph, shader_texture_Node, "out", shader_finalize_Node, "specTrans");
                else if(key == "Emission")
                    Zeno_AddLink(hGraph, shader_texture_Node, "out", shader_finalize_Node, "emission");
                else if(key == "Emission Strength")
                    Zeno_AddLink(hGraph, shader_texture_Node, "out", shader_finalize_Node, "emissionIntensity");

                if(key == "Normal")
                    Zeno_AddLink(hGraph, shader_texture_Node, "out", shader_finalize_Node, "normal");

                shader_texture_count++;
            }

            ZENO_HANDLE make_list_Node = Zeno_AddNode(hGraph, "MakeList");
            Zeno_SetPos(hGraph, make_list_Node, NodePos1);

            int texture_2d_count = 0;
            for(auto& tex_path :mat_texs){
                std::pair<float, float> NodePos_1 = { NodePos1.first - 700.0f, NodePos1.second + 500.0f * texture_2d_count};

                ZENO_HANDLE texture_2d_Node = Zeno_AddNode(hGraph, "MakeTexture2D");
                Zeno_SetPos(hGraph, texture_2d_Node, NodePos_1);

                Zeno_AddLink(hGraph, texture_2d_Node, "tex", make_list_Node, "obj"+std::to_string(0));

                Zeno_SetInputDefl(hGraph, texture_2d_Node, "path", tex_path);

                texture_2d_count++;
            }

            Zeno_AddLink(hGraph, make_list_Node, "list", shader_finalize_Node, "tex2dList");
        }

        mat_count++;
    }
}

void EvalBlenderFile::onExecClicked() {
    auto inputs = GetCurNodeInputs();
    std::string script_file = inputs[0];
    std::string blender_file = inputs[1];
    std::string output_path = inputs[2];

    std::string command = "python \"" + script_file + "\" -i \"" + blender_file + "\" -o \"" + output_path + "\"";
    std::cout << "eval blender - command: " << command << "\n";
    std::system(command.c_str());
}

std::vector<std::string> EvalBlenderFile::GetCurNodeInputs() {
    ZENO_HANDLE hGraph = Zeno_GetGraph("main");

    ZENO_HANDLE evalBlenderNode = index().internalId();

    // Get Node Paths
    ZVARIANT _script_file;
    ZVARIANT _blender_file;
    ZVARIANT _output_path;
    std::string type;
    Zeno_GetInputDefl(hGraph, evalBlenderNode, "script_file", _script_file, type);
    Zeno_GetInputDefl(hGraph, evalBlenderNode, "blender_file", _blender_file, type);
    Zeno_GetInputDefl(hGraph, evalBlenderNode, "output_path", _output_path, type);
    std::string script_file = std::get<std::string>(_script_file);
    std::string blender_file = std::get<std::string>(_blender_file);
    std::string output_path = std::get<std::string>(_output_path);

    std::vector<std::string> inputs{};
    inputs.emplace_back(script_file);
    inputs.emplace_back(blender_file);
    inputs.emplace_back(output_path);
    return inputs;
}
