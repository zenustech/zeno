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

using Path = std::filesystem::path;

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
