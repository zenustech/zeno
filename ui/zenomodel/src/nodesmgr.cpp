#include "nodesmgr.h"
#include <zenomodel/include/uihelper.h>
#include <zeno/utils/log.h>
#include <zenomodel/include/graphsmanagment.h>
#include <zenomodel/include/curvemodel.h>
#include "variantptr.h"
#include <zenomodel/include/curveutil.h>
#include "zassert.h"
#include "graphsmodel.h"


QString NodesMgr::createNewNode(IGraphsModel* pModel, QModelIndex subgIdx, const QString& descName, const QPointF& pt)
{
    zeno::log_debug("onNewNodeCreated");
    NODE_DATA node = newNodeData(pModel, descName, pt);
    pModel->addNode(node, subgIdx, true);
    return node.ident;
}

QString NodesMgr::createExtractDictNode(IGraphsModel *pModel, QModelIndex subgIdx, const QString &infos) 
{
    zeno::log_debug("onExtractDictCreated");
    NODE_DATA node = newNodeData(pModel, "ExtractDict", QPointF());
    if (!infos.isEmpty()) {
        QStringList lst = infos.split(",");
        for (auto name : lst) {
            OUTPUT_SOCKET newSocket;
            newSocket.info.name = name;
            newSocket.info.control = CONTROL_NONE;
            newSocket.info.sockProp = SOCKPROP_EDITABLE;
            node.outputs.insert(name, newSocket);
        }
    }
    pModel->addNode(node, subgIdx, true);
    return node.ident;
}

NODE_DATA NodesMgr::newNodeData(IGraphsModel* pModel, const QString& descName, const QPointF& pt)
{
    NODE_DATA node;
    NODE_DESC desc;
    bool ret = pModel->getDescriptor(descName, desc);
    ZASSERT_EXIT(ret, node);

    const QString &nodeid = UiHelper::generateUuid(descName);
    node.ident = nodeid;
    node.nodeCls = descName;
    node.type = nodeType(descName);
    initInputSocks(pModel, nodeid, node.inputs, desc.is_subgraph);
    initOutputSocks(pModel, nodeid, node.outputs);
    initParams(descName, pModel, node.params);
    node.parmsNotDesc = initParamsNotDesc(descName);
    node.pos = pt;
    node.bCollasped = false;
    return node;
}

NODE_TYPE NodesMgr::nodeType(const QString& name)
{
    if (name == "Blackboard")
    {
        return BLACKBOARD_NODE;
    }
    else if (name == "Group")
    {
        return GROUP_NODE;
    }
    else if (name == "SubInput")
    {
        return SUBINPUT_NODE;
    }
    else if (name == "SubOutput")
    {
        return SUBOUTPUT_NODE;
    }
    else if (name == "MakeHeatmap")
    {
        return HEATMAP_NODE;
    }
    else
    {
        return NORMAL_NODE;
    }
}

void NodesMgr::initInputSocks(IGraphsModel* pGraphsModel, const QString& nodeid, INPUT_SOCKETS& descInputs, bool isSubgraph)
{
    if (descInputs.find("SRC") == descInputs.end())
    {
        INPUT_SOCKET srcSocket;
        srcSocket.info.name = "SRC";
        srcSocket.info.control = CONTROL_NONE;
        srcSocket.info.nodeid = nodeid;
        descInputs.insert("SRC", srcSocket);
    }
}

void NodesMgr::initOutputSocks(IGraphsModel* pModel, const QString& nodeid, OUTPUT_SOCKETS& descOutputs)
{
    if (descOutputs.find("DST") == descOutputs.end())
    {
        OUTPUT_SOCKET dstSocket;
        dstSocket.info.name = "DST";
        dstSocket.info.control = CONTROL_NONE;
        dstSocket.info.nodeid = nodeid;
        descOutputs.insert("DST", dstSocket);
    }

    if (descOutputs.lastKey() != "DST")
    {
        //ensure that the "DST" is the last key in sockets.
        OUTPUT_SOCKET dstSocket = descOutputs["DST"];
        descOutputs.remove("DST");
        descOutputs.insert("DST", dstSocket);
    }
}

void NodesMgr::initParams(const QString& descName, IGraphsModel* pGraphsModel, PARAMS_INFO& params)
{
    if (params.find("curve") != params.end())
    {
        PARAM_INFO& param = params["curve"];
        if (param.control == CONTROL_CURVE)
        {
            CURVES_MODEL curves;
            QString ids[] = {"x", "y", "z"};
            for (int i = 0; i < 3; i++) {
                CurveModel *pModel = curve_util::deflModel(pGraphsModel);
                pModel->setData(pModel->index(0, 0), QVariant::fromValue(QPointF(0, i*0.5)), ROLE_NODEPOS);
                pModel->setData(pModel->index(1, 0), QVariant::fromValue(QPointF(1, 1-i*0.5)), ROLE_NODEPOS);
                pModel->setId(ids[i]);
                curves.insert(ids[i], pModel);
            }
            param.value = QVariant::fromValue(curves);
        }
    }
    if (descName == "MakeHeatmap" && params.find("_RAMPS") == params.end())
    {
        PARAM_INFO param;
        param.control = CONTROL_COLOR;
        param.name = "_RAMPS";
        param.bEnableConnect = false;
        QLinearGradient grad;
        grad.setColorAt(0, QColor::fromRgbF(0., 0., 0.));
        grad.setColorAt(1, QColor::fromRgbF(1., 1., 1.));
        param.value = UiHelper::gradient2colorString(grad);
        params.insert(param.name, param);
    }
    if (descName == "SubInput" || descName == "SubOutput")
    {
        ZASSERT_EXIT(params.find("name") != params.end() &&
                     params.find("type") != params.end() &&
                     params.find("defl") != params.end());
        const QVariant& defl = params["defl"].value.toString();
        const QString& typeDesc = params["type"].value.toString();
        if (typeDesc.isEmpty())
        {
            params["defl"].typeDesc = "";
            params["defl"].control = CONTROL_NONE;
        }
    }
}

PARAMS_INFO NodesMgr::initParamsNotDesc(const QString& name)
{
    PARAMS_INFO paramsNotDesc;
    if (name == "Blackboard" || name == "Group")
    {
        BLACKBOARD_INFO blackboard;
        blackboard.content = tr("content");
        blackboard.title = tr("title");
        blackboard.background = QColor(0, 100, 168);
        blackboard.sz = QSize(500, 500);
        paramsNotDesc["blackboard"].name = "blackboard";
        paramsNotDesc["blackboard"].value = QVariant::fromValue(blackboard);
    }
    return paramsNotDesc;
}
