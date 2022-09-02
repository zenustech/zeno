#include "nodesmgr.h"
#include "fuzzy_search.h"
#include <zenoui/util/uihelper.h>
#include "util/apphelper.h"
#include <zeno/utils/log.h>
#include "zenoapplication.h"
#include "graphsmanagment.h"
#include <zenoui/model/curvemodel.h>
#include <zenoui/model/variantptr.h>
#include "curvemap/curveutil.h"


void NodesMgr::createNewNode(IGraphsModel* pModel, QModelIndex subgIdx, const QString& descName, const QPointF& pt)
{
    zeno::log_debug("onNewNodeCreated");
    NODE_DESCS descs = pModel->descriptors();
    NODE_DESC desc = descs[descName];

    const QString& nodeid = UiHelper::generateUuid(descName);
    NODE_DATA node;
    node[ROLE_OBJID] = nodeid;
    node[ROLE_OBJNAME] = descName;
    node[ROLE_NODETYPE] = nodeType(descName);
    initInputSocks(pModel, nodeid, desc.inputs);
    node[ROLE_INPUTS] = QVariant::fromValue(desc.inputs);
    initOutputSocks(pModel, nodeid, desc.outputs);
    node[ROLE_OUTPUTS] = QVariant::fromValue(desc.outputs);
    initParams(descName, pModel, desc.params);
    node[ROLE_PARAMETERS] = QVariant::fromValue(desc.params);
    node[ROLE_PARAMS_NO_DESC] = QVariant::fromValue(initParamsNotDesc(descName));
    node[ROLE_OBJPOS] = pt;
    node[ROLE_COLLASPED] = false;

    pModel->addNode(node, subgIdx, true);
}

NODE_TYPE NodesMgr::nodeType(const QString& name)
{
    if (name == "Blackboard")
    {
        return BLACKBOARD_NODE;
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

void NodesMgr::initInputSocks(IGraphsModel* pGraphsModel, const QString& nodeid, INPUT_SOCKETS& descInputs)
{
    if (descInputs.find("SRC") == descInputs.end())
    {
        INPUT_SOCKET srcSocket;
        srcSocket.info.name = "SRC";
        srcSocket.info.control = CONTROL_NONE;
        srcSocket.info.nodeid = nodeid;
        descInputs.insert("SRC", srcSocket);
    }

    if (descInputs.lastKey() != "SRC")
    {
        //ensure that the "SRC" is the last key in sockets.
        INPUT_SOCKET srcSocket = descInputs["SRC"];
        descInputs.remove("SRC");
        descInputs.insert("SRC", srcSocket);
    }

    if (descInputs.find("curve") != descInputs.end())
    {
        INPUT_SOCKET& input = descInputs["curve"];
        if (input.info.control == CONTROL_CURVE)
        {
            CurveModel *pModel = curve_util::deflModel(pGraphsModel);
            input.info.defaultValue = QVariantPtr<CurveModel>::asVariant(pModel);
        }
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
            CurveModel *pModel = curve_util::deflModel(pGraphsModel);
            param.value = QVariantPtr<CurveModel>::asVariant(pModel);
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
        param.value = AppHelper::gradient2colorString(grad);
        params.insert(param.name, param);
    }
}

PARAMS_INFO NodesMgr::initParamsNotDesc(const QString& name)
{
    PARAMS_INFO paramsNotDesc;
    if (name == "Blackboard")
    {
        BLACKBOARD_INFO blackboard;
        blackboard.content = tr("Please input the content of blackboard");
        blackboard.title = tr("Please input the title of blackboard");
        paramsNotDesc["blackboard"].name = "blackboard";
        paramsNotDesc["blackboard"].value = QVariant::fromValue(blackboard);
    }
    return paramsNotDesc;
}
