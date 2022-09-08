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

ReadFBXPrim::ReadFBXPrim(const NodeUtilParam& params, QGraphicsItem* parent)
    : ZenoNode(params, parent)
{

}

ReadFBXPrim::~ReadFBXPrim()
{

}

QGraphicsLinearLayout* ReadFBXPrim::initCustomParamWidgets()
{
    QGraphicsLinearLayout* pHLayout = new QGraphicsLinearLayout(Qt::Horizontal);

    ZenoTextLayoutItem* pNameItem = new ZenoTextLayoutItem("node", m_renderParams.paramFont, m_renderParams.paramClr.color());
    pHLayout->addItem(pNameItem);

    ZenoParamPushButton* pEditBtn = new ZenoParamPushButton("Generate", -1, QSizePolicy::Expanding);
    pHLayout->addItem(pEditBtn);
    connect(pEditBtn, SIGNAL(clicked()), this, SLOT(onEditClicked()));

    return pHLayout;
}

void ReadFBXPrim::onEditClicked()
{
    zeno::log_info("ReadFBXPrim Generate Nodes");
    ZENO_HANDLE hGraph = Zeno_GetGraph("main");

    // Get ReadFBXPrim ident
    ZENO_HANDLE fbxNode = index().internalId();
    std::string fbxIdentStr;
    Zeno_GetIdent(fbxNode, fbxIdentStr);

    // Find corresponding node
    auto &inst = Zenovis::GetInstance();
    auto sess = inst.getSession();
    ZASSERT_EXIT(sess);
    auto scene = sess->get_scene();
    ZASSERT_EXIT(scene);
    std::vector<std::pair<std::string, zeno::IObject *>> const &objs
        = scene->objectsMan->pairs();
    zeno::IObject* fbxObj;
    bool findFbxObj = false;
    for (auto const &[key, obj] : objs) {
        if(key.find(fbxIdentStr, 0) != std::string::npos){
            findFbxObj = true;
            fbxObj = obj;
        }
    }

    // Get the position of current ReadFBXPrim node
    std::pair<float, float> fbxNodePos;
    Zeno_GetPos(fbxNode, fbxNodePos);

    // Create nodes
    if(findFbxObj) {
        auto matNum = fbxObj->userData().getLiterial<int>("matNum");
        auto fbxName = fbxObj->userData().getLiterial<std::string>("fbxName");
        ZENO_HANDLE fbxPartGraph = Zeno_GetGraph("FBXPart");
        ZASSERT_EXIT(fbxPartGraph);

        for (int i = 0; i < matNum; i++) {
            auto matName = fbxObj->userData().getLiterial<std::string>(std::to_string(i));
            zeno::log_info("Create with mat name {}, fbx name {}", matName, fbxName);

            ZENO_HANDLE dictNode = Zeno_AddNode(hGraph, "DictGetItem");
            std::pair<float, float> dictNodePos = {fbxNodePos.first + 500.0f, fbxNodePos.second + i * 300.0f};
            Zeno_SetPos(dictNode, dictNodePos);

            Zeno_SetInputDefl(dictNode, "key", matName);
            Zeno_AddLink(fbxNode, "mats", dictNode, "dict");

            ZENO_HANDLE forkedSubg = 0;
            ZENO_HANDLE forkedNode = 0;
            std::string fbxPartGraphName = fbxName+"_"+matName;
            ZENO_ERROR ret = Zeno_ForkGraph(hGraph, "FBXPart", forkedSubg, forkedNode);
            ZASSERT_EXIT(!ret);ZASSERT_EXIT(forkedSubg);ZASSERT_EXIT(forkedNode);
            Zeno_RenameGraph(forkedSubg, fbxPartGraphName);

            std::pair<float, float> fbxPartPos = {dictNodePos.first + 500.0f, dictNodePos.second};
            ZENO_HANDLE newFbxPartNode = Zeno_AddNode(hGraph, fbxPartGraphName);
            ZASSERT_EXIT(newFbxPartNode);

            Zeno_SetPos(newFbxPartNode, fbxPartPos);
            Zeno_AddLink(dictNode, "object", newFbxPartNode, "data");
            Zeno_SetView(newFbxPartNode, true);
        }

        Zeno_SetView(fbxNode, false);
    }else{
        zeno::log_error("Not found ReadFBXPrim node in objectsMan");
    }
}