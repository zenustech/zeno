#include "readfbxprim.h"
#include "util/log.h"
#include <zenomodel/include/api.h>
#include <zenomodel/include/igraphsmodel.h>
#include "zenoapplication.h"
#include <zenomodel/include/graphsmanagment.h>

#include <viewport/zenovis.h>
#include <zeno/core/Session.h>
#include <zeno/extra/ObjectsManager.h>
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
    ZENO_HANDLE mGraph = Zeno_GetGraph("main");


    auto inputs = GetCurNodeInputs();
    std::string output_path = inputs[2];
    Path dir (output_path);
    Path info_file_path ("info.json");
    Path map_file_path ("map.json");
    Path abc_file_path ("output.abc");
    info_file_path = dir / info_file_path;
    map_file_path = dir / map_file_path;
    abc_file_path = dir / abc_file_path;


    bool use_map_file = false;

    Json mat_map;
    std::ifstream map_file(map_file_path.string());
    std::string map_str;
    if(map_file) {
        std::ostringstream ss;
        ss << map_file.rdbuf(); // reading data
        map_str = ss.str();
        use_map_file = true;
        mat_map = Json::parse(map_str);
    }else{
        zeno::log_warn("Can not open map.json under folder: {}\n",dir);
    }

    std::ifstream info_file(info_file_path.string());
    std::string info_str;
    if(info_file) {
        std::ostringstream ss;
        ss << info_file.rdbuf(); // reading data
        info_str = ss.str();
    }else{
        zeno::log_error("Can not open info.json under folder: {}\n",dir);
        return;
    }
    Json mat_info = Json::parse(info_str);
    // Eval Geo
    std::pair<float,float> pos;
    ZENO_HANDLE GeoSubG = Zeno_GetGraph("AlembicImport");
    if(GeoSubG==0){
        GeoSubG = Zeno_CreateGraph("AlembicImport");
        ZENO_HANDLE BindMat = Zeno_GetGraph("FacesetBindMat");
        if(BindMat==0){
            BindMat = Zeno_CreateGraph("FacesetBindMat");

            ZENO_HANDLE input = Zeno_AddNode(BindMat,"SubInput");
            Zeno_SetParam(BindMat,input,"name","abcPrim");

            ZENO_HANDLE AlembicSplitByName = Zeno_AddNode(BindMat,"AlembicSplitByName");

            ZENO_HANDLE BeginForEach = Zeno_AddNode(BindMat,"BeginForEach");

            ZENO_HANDLE DictGetItem = Zeno_AddNode(BindMat,"DictGetItem");

            ZENO_HANDLE PrimitiveTriangulate = Zeno_AddNode(BindMat,"PrimitiveTriangulate");
            Zeno_SetParam(BindMat,PrimitiveTriangulate,"from_poly",true);
            Zeno_SetParam(BindMat,PrimitiveTriangulate,"with_uv",true);

            ZENO_HANDLE PrimitiveCalcNormal = Zeno_AddNode(BindMat,"PrimitiveCalcNormal");

            ZENO_HANDLE BindMaterial = Zeno_AddNode(BindMat,"BindMaterial");

            ZENO_HANDLE EndForEach = Zeno_AddNode(BindMat,"EndForEach");

            ZENO_HANDLE ouput = Zeno_AddNode(BindMat,"SubOutput");
            Zeno_SetParam(BindMat,ouput,"name","list");

            Zeno_AddLink(BindMat,input,"port",AlembicSplitByName,"prim");
            Zeno_AddLink(BindMat,AlembicSplitByName,"dict",DictGetItem,"dict");
            Zeno_AddLink(BindMat,AlembicSplitByName,"namelist",BeginForEach,"list");
            Zeno_AddLink(BindMat,BeginForEach,"object",DictGetItem,"key");
            Zeno_AddLink(BindMat,BeginForEach,"object",BindMaterial,"mtlid");
            Zeno_AddLink(BindMat,BeginForEach,"FOR",EndForEach,"FOR");
            Zeno_AddLink(BindMat,DictGetItem,"object",PrimitiveTriangulate,"prim");
            Zeno_AddLink(BindMat,PrimitiveTriangulate,"prim",PrimitiveCalcNormal,"prim");
            Zeno_AddLink(BindMat,PrimitiveCalcNormal,"prim",BindMaterial,"object");
            Zeno_AddLink(BindMat,BindMaterial,"object",EndForEach,"object");
            Zeno_AddLink(BindMat,EndForEach,"list",ouput,"port");

        }
        ZENO_HANDLE input = Zeno_AddNode(GeoSubG,"SubInput");
        Zeno_SetParam(GeoSubG,input,"name","abc_file_path");
        Zeno_SetParam(GeoSubG,input,"type","string");
        ZENO_HANDLE ReadAlembic = Zeno_AddNode(GeoSubG,"ReadAlembic");
        Zeno_SetInputDefl(GeoSubG,ReadAlembic,"path",abc_file_path.string());
        Zeno_SetInputDefl(GeoSubG,ReadAlembic,"read_face_set",true);

        ZENO_HANDLE AlembicPrimList = Zeno_AddNode(GeoSubG,"AlembicPrimList");
        Zeno_SetInputDefl(GeoSubG,AlembicPrimList,"use_xform",true);
        Zeno_SetOnce(GeoSubG,AlembicPrimList,true);

        ZENO_HANDLE CountAlembicPrims = Zeno_AddNode(GeoSubG,"CountAlembicPrims");
        ZENO_HANDLE BeginFor = Zeno_AddNode(GeoSubG,"BeginFor");
        ZENO_HANDLE ListGetItem = Zeno_AddNode(GeoSubG,"ListGetItem");
        ZENO_HANDLE FacesetBindMat = Zeno_AddNode(GeoSubG,"FacesetBindMat");
        ZENO_HANDLE EndForEach = Zeno_AddNode(GeoSubG,"EndForEach");
        ZENO_HANDLE output = Zeno_AddNode(GeoSubG,"SubOutput");

        Zeno_AddLink(GeoSubG,input,"port",ReadAlembic,"path");
        Zeno_AddLink(GeoSubG,ReadAlembic,"abctree",AlembicPrimList,"abctree");
        Zeno_AddLink(GeoSubG,ReadAlembic,"abctree",CountAlembicPrims,"abctree");
        Zeno_AddLink(GeoSubG,AlembicPrimList,"prims",ListGetItem,"list");
        Zeno_AddLink(GeoSubG,ListGetItem,"object",FacesetBindMat,"abcPrim");
        Zeno_AddLink(GeoSubG,FacesetBindMat,"list",EndForEach,"list");
        Zeno_AddLink(GeoSubG,CountAlembicPrims,"count",BeginFor,"count");
        Zeno_AddLink(GeoSubG,BeginFor,"index",ListGetItem,"index");
        Zeno_AddLink(GeoSubG,BeginFor,"FOR",EndForEach,"FOR");
        Zeno_AddLink(GeoSubG,EndForEach,"list",output,"port");
        

    }

    ZENO_HANDLE import_node = Zeno_AddNode(mGraph,"AlembicImport");
    Zeno_SetInputDefl(mGraph,import_node,"abc_file_path",abc_file_path.string());

    Zeno_SetView(mGraph,import_node,true);
    Zeno_GetPos(mGraph,import_node,pos);
    pos.second += 500;
    

    

    // Eval Material

    int mat_num = 0;
    for(auto& [key,val]:mat_info.items()){
        mat_num++;
    }
    
    int col = std::sqrt(mat_num);
    int index = 0;
    float h_start = pos.first;
    for (auto& [key, val] : mat_info.items()){
        if(index >= col){
            index = 0;
            pos.first = h_start;
            pos.second += 200;
        }else{
            index ++;
            pos.first += 500;
        }

        auto matName = key;

        ZENO_HANDLE subG = Zeno_GetGraph(matName);
        if(subG != 0){
            zeno::log_warn("Material SubGraph {} already exist! Please remove it then run again");
            ZENO_HANDLE mat_subgraph = Zeno_AddNode(mGraph,matName);
            Zeno_SetView(mGraph,mat_subgraph,true);
            Zeno_SetPos(mGraph,mat_subgraph,pos);
            continue;
        }
        subG = Zeno_CreateGraph(matName,1);
        ZENO_HANDLE shader = Zeno_AddNode(subG,"ShaderFinalize");

        ZENO_HANDLE sub_output= Zeno_AddNode(subG, "SubOutput");
        Zeno_AddLink(subG, shader, "mtl", sub_output, "port");
        Zeno_SetInputDefl(subG,shader,"mtlid",matName);
        for (auto& [mat_item,mat_item_info]: val.items()){

            std::string item_name = mat_item;
            if(use_map_file && mat_map.find(item_name)!=mat_map.end()){
                item_name = mat_map.at(item_name).get<std::string>();
            }

            ZVARIANT ret;
            std::string type = "";
            if(Zeno_GetInputDefl(subG,shader,item_name,ret,type)==ErrorCode::Err_SockNotExist){ //if no such a material item name, jump to next item
                zeno::log_warn("No such a material item: {} in zeno, import from material: {}",item_name,matName);
                continue;
            }

            if(mat_item_info.find("path")!=mat_item_info.end()){ //has texture input for this item
                ZENO_HANDLE texture_node = Zeno_AddNode(subG,"SmartTexture2D");
                Zeno_SetInputDefl(subG,texture_node,"path",mat_item_info.at("path").get<std::string>());
                Zeno_SetInputDefl(subG,texture_node,"type",mat_item_info.at("channel").get<std::string>());
                if(type == "float"&&   mat_item_info.at("channel").get<std::string>() == "vec3"){
                    Zeno_SetInputDefl(subG,texture_node,"type","float");
                }
                Zeno_SetInputDefl(subG,texture_node,"post_process",mat_item_info.at("postpro").get<std::string>());
                if(item_name == "normal"){
                    Zeno_SetInputDefl(subG,texture_node,"value",zeno::vec4f(0.0f,0.0f,1.0f,0.0f));
                }
                Zeno_AddLink(subG,texture_node,"out",shader,item_name);
            }else{
                std::vector<float> value;
                if(mat_item_info.at("value").is_array()){
                    value = mat_item_info.at("value").get<std::vector<float>>();
                }else{
                    value.push_back(mat_item_info.at("value").get<float>());
                }
                if(item_name == "normal" && std::inner_product(value.begin(),value.end(),value.begin(),0) == 0 ){
                    value[0] = 0.0f;
                    value[1] = 0.0f;
                    value[2] = 1.0f;
                }

                int input_size = value.size();
                int need_size = 3;
                if(type == "vec3f" || type == "colorvec3f"){
                    if (input_size >= 3 ){
                        Zeno_SetInputDefl(subG,shader,item_name,zeno::vec3f(value[0],value[1],value[2]));
                    }else{
                        Zeno_SetInputDefl(subG,shader,item_name,zeno::vec3f(value[0],value[0],value[0])); 
                    }
                }else if(type == "float"){
                    Zeno_SetInputDefl(subG,shader,item_name,value[0]);
                }

            }

        }
        
        ZENO_HANDLE mat_subgraph = Zeno_AddNode(mGraph,matName);
        Zeno_SetView(mGraph,mat_subgraph,true);
        Zeno_SetPos(mGraph,mat_subgraph,pos);

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
