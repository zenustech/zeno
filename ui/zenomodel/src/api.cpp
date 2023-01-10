#include "api.h"
#include "graphsmanagment.h"
#include "modelrole.h"
#include "nodesmgr.h"
#include "apiutil.h"
#include "zeno/utils/log.h"
#include "variantptr.h"
#include "nodeparammodel.h"
#include "uihelper.h"


ZENO_ERROR Zeno_NewFile()
{
    IGraphsModel* pModel = GraphsManagment::instance().newFile();
    return pModel != nullptr;
}

ZENO_HANDLE Zeno_CreateGraph(const std::string& name)
{
    IGraphsModel *pModel = GraphsManagment::instance().currentModel();
    if (!pModel)
        return Err_ModelNull;

    const QString& qsName = QString::fromStdString(name);
    pModel->newSubgraph(QString::fromStdString(name));
    QModelIndex subgIdx = pModel->index(qsName);
    return subgIdx.internalId();
}

ZENO_ERROR Zeno_DeleteGraph(ZENO_HANDLE hSubgraph)
{
    IGraphsModel* pModel = GraphsManagment::instance().currentModel();
    if (!pModel)
        return Err_ModelNull;

    QModelIndex subgIdx = pModel->subgIndex(hSubgraph);
    pModel->removeSubGraph(subgIdx.data(ROLE_OBJNAME).toString());
    return Err_NoError;
}

ZENO_HANDLE Zeno_GetGraph(const std::string &name)
{
    IGraphsModel* pModel = GraphsManagment::instance().currentModel();
    if (!pModel)
        return Err_ModelNull;

    QModelIndex subgIdx = pModel->index(QString::fromStdString(name));
    return subgIdx.internalId();
}

ZENO_ERROR Zeno_RenameGraph(ZENO_HANDLE hSubgraph, const std::string& newName)
{
    IGraphsModel* pModel = GraphsManagment::instance().currentModel();
    if (!pModel)
        return Err_ModelNull;

    QModelIndex subgIdx = pModel->subgIndex(hSubgraph);
    const QString& subgName = subgIdx.data(ROLE_OBJNAME).toString();
    pModel->renameSubGraph(subgName, QString::fromStdString(newName));
    return Err_NoError;
}

ZENO_ERROR  Zeno_ForkGraph(
        ZENO_HANDLE hSubgWhere,
        const std::string& name,
        ZENO_HANDLE& hForkedSubg,
        ZENO_HANDLE& hForkedNode
)
{
    IGraphsModel* pModel = GraphsManagment::instance().currentModel();
    if (!pModel)
        return Err_ModelNull;

    QModelIndex subgIdx = pModel->subgIndex(hSubgWhere);
    if (!subgIdx.isValid()) {
        return Err_SubgNotExist;
    }

    QModelIndex toForkIdx = pModel->index(QString::fromStdString(name));
    if (!toForkIdx.isValid()) {
        return Err_SubgNotExist;
    }

    QModelIndex newForkNode = pModel->fork(subgIdx, toForkIdx);
    const QString& newSubgName = newForkNode.data(ROLE_OBJNAME).toString();
    QModelIndex newSubgIdx = pModel->index(newSubgName);

    hForkedSubg = newSubgIdx.internalId();
    hForkedNode = newForkNode.internalId();
    return Err_NoError;
}

int Zeno_GetCount()
{
    IGraphsModel* pModel = GraphsManagment::instance().currentModel();
    if (!pModel)
        return Err_ModelNull;
    return pModel->rowCount();
}

ZENO_HANDLE Zeno_GetItem(int idx)
{
    IGraphsModel* pModel = GraphsManagment::instance().currentModel();
    if (!pModel)
        return Err_ModelNull;

    QModelIndex subgIdx = pModel->index(idx, 0);
    return subgIdx.internalId();
}

ZENO_HANDLE Zeno_AddNode(ZENO_HANDLE hGraph, const std::string& nodeCls)
{
    IGraphsModel* pModel = GraphsManagment::instance().currentModel();
    if (!pModel)
        return Err_ModelNull;

    QModelIndex subgIdx = pModel->subgIndex(hGraph);
    if (!subgIdx.isValid())
        return -1;

    QString ident = NodesMgr::createNewNode(pModel, subgIdx, QString::fromStdString(nodeCls), QPointF(0, 0));
    return pModel->index(ident, subgIdx).internalId();
}

ZENO_ERROR Zeno_DeleteNode(ZENO_HANDLE hNode)
{
    IGraphsModel *pModel = GraphsManagment::instance().currentModel();
    if (!pModel)
        return Err_ModelNull;

    QModelIndex subgIdx = pModel->subgByNodeId(hNode);
    QModelIndex nodeIdx = pModel->nodeIndex(hNode);
    pModel->removeNode(nodeIdx.data(ROLE_OBJID).toString(), subgIdx);
    return Err_NoError;
}

ZENO_ERROR Zeno_GetName(ZENO_HANDLE hNode, std::string& name)
{
    IGraphsModel *pModel = GraphsManagment::instance().currentModel();
    if (!pModel)
        return Err_ModelNull;

    QModelIndex idx = pModel->nodeIndex(hNode);
    name = idx.data(ROLE_OBJNAME).toString().toStdString();
    return Err_NoError;
}

//io
ZENO_ERROR Zeno_OpenFile(const std::string &fn)
{
    IGraphsModel *pModel = GraphsManagment::instance().openZsgFile(QString::fromStdString(fn));
    if (!pModel)
        return Err_IOError;
    return Err_NoError;
}

ZENO_ERROR Zeno_SaveAs(const std::string &fn)
{
    APP_SETTINGS settings;
    bool ret = GraphsManagment::instance().saveFile(QString::fromStdString(fn), settings);
    return ret ? Err_NoError : -1;
}

ZENO_ERROR Zeno_AddLink(ZENO_HANDLE hOutnode, const std::string &outSock,
                        ZENO_HANDLE hInnode, const std::string &inSock)
{
    IGraphsModel* pModel = GraphsManagment::instance().currentModel();
    if (!pModel)
        return Err_ModelNull;

    QModelIndex outIdx = pModel->nodeIndex(hOutnode);
    if (!outIdx.isValid()) {
        zeno::log_error("miss hOutnode: {}", hOutnode);
        return Err_NodeNotExist;
    }
    OUTPUT_SOCKETS outputs = outIdx.data(ROLE_OUTPUTS).value<OUTPUT_SOCKETS>();
    const QString qsOutSock = QString::fromStdString(outSock);
    if (!outputs.contains(qsOutSock)) {
        zeno::log_error("miss outSock: {}", outSock);
        return Err_SockNotExist;
    }

    QModelIndex inIdx = pModel->nodeIndex(hInnode);
    if (!inIdx.isValid()) {
        zeno::log_error("miss hInnode: {}", hInnode);
        return Err_NodeNotExist;
    }
    INPUT_SOCKETS inputs = inIdx.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
    const QString qsInSock = QString::fromStdString(inSock);
    if (!inputs.contains(qsInSock)) {
        zeno::log_error("miss inSock: {}", inSock);
        return Err_SockNotExist;
    }

    //get subgraph directly from node.
    QModelIndex subgIdx = pModel->subgByNodeId(hInnode);

    NodeParamModel* inNodeParams = QVariantPtr<NodeParamModel>::asPtr(inIdx.data(ROLE_NODE_PARAMS));
    QModelIndex inSockIdx = inNodeParams->getParam(PARAM_INPUT, QString::fromStdString(inSock));

    NodeParamModel* outNodeParams = QVariantPtr<NodeParamModel>::asPtr(outIdx.data(ROLE_NODE_PARAMS));
    QModelIndex outSockIdx = outNodeParams->getParam(PARAM_OUTPUT, QString::fromStdString(outSock));

    pModel->addLink(outSockIdx, inSockIdx);
    return Err_NoError;
}

ZENO_ERROR Zeno_RemoveLink(ZENO_HANDLE hOutnode, const std::string& outSock,
                           ZENO_HANDLE hInnode, const std::string &inSock)
{
    IGraphsModel* pModel = GraphsManagment::instance().currentModel();
    if (!pModel)
        return Err_ModelNull;

    QModelIndex outIdx = pModel->nodeIndex(hOutnode);
    if (!outIdx.isValid())
        return Err_NodeNotExist;

    QModelIndex inIdx = pModel->nodeIndex(hInnode);
    if (!inIdx.isValid())
        return Err_NodeNotExist;

    QModelIndex linkIdx = pModel->linkIndex(outIdx.data(ROLE_OBJID).toString(),
                                            QString::fromStdString(outSock),
                                            inIdx.data(ROLE_OBJID).toString(),
                                            QString::fromStdString(inSock));
    QModelIndex subgIdx = pModel->subgByNodeId(hInnode);

    pModel->removeLink(linkIdx);
    return Err_NoError;
}

ZENO_ERROR Zeno_GetOutNodes(
        ZENO_HANDLE hNode,
        const std::string& outSock,
        std::vector<std::pair<ZENO_HANDLE, std::string>>& res)
{
    IGraphsModel* pModel = GraphsManagment::instance().currentModel();
    if (!pModel)
        return Err_ModelNull;

    QModelIndex idx = pModel->nodeIndex(hNode);
    if (!idx.isValid())
        return Err_NodeNotExist;

    QModelIndex subgIdx = pModel->subgByNodeId(hNode);
    if (!subgIdx.isValid())
        return Err_SubgNotExist;

    OUTPUT_SOCKETS outputs = idx.data(ROLE_OUTPUTS).value<OUTPUT_SOCKETS>();
    const QString& qsOutSock = QString::fromStdString(outSock);
    if (outputs.find(qsOutSock) == outputs.end())
    {
        return Err_SockNotExist;
    }

    OUTPUT_SOCKET output = outputs[qsOutSock];
    for (auto link : output.info.links)
    {
        QString inNode = UiHelper::getSockNode(link.inSockPath);
        QModelIndex inIdx = pModel->index(inNode, subgIdx);
        ZENO_HANDLE hdl = inIdx.internalId();
        res.push_back(std::make_pair(hdl, inNode.toStdString()));
    }

    return Err_NoError;
}

ZENO_ERROR Zeno_GetInput(
        ZENO_HANDLE hNode,
        const std::string& inSock,
        std::pair<ZENO_HANDLE, std::string>& ret)
{
    IGraphsModel *pModel = GraphsManagment::instance().currentModel();
    if (!pModel)
        return Err_ModelNull;

    QModelIndex idx = pModel->nodeIndex(hNode);
    if (!idx.isValid())
        return Err_NodeNotExist;

    INPUT_SOCKETS inputs = idx.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
    const QString& qsInSock = QString::fromStdString(inSock);
    if (inputs.find(qsInSock) == inputs.end())
    {
        return Err_SockNotExist;
    }

    const QString& nodeid = idx.data(ROLE_OBJID).toString();
    const QModelIndex& subgIdx = pModel->subgByNodeId(hNode);
    INPUT_SOCKET input = inputs[qsInSock];

    if (1 == input.info.links.size())
    {
        EdgeInfo link = input.info.links[0];

        QString outNode = UiHelper::getSockNode(link.outSockPath);
        QString outSock = UiHelper::getSockName(link.outSockPath);
        QModelIndex outIdx = pModel->index(outNode, subgIdx);
        ret.first = outIdx.internalId();
        ret.second = outSock.toStdString();
        return Err_NoError;
    }
    else
    {
        return Err_NoConnection;
    }
}

ZENO_ERROR Zeno_GetInputDefl(
        ZENO_HANDLE hNode,
        const std::string& inSock,
        /*out*/ ZVARIANT& ret,
        /*out*/ std::string &type)
{
    IGraphsModel *pModel = GraphsManagment::instance().currentModel();
    if (!pModel)
        return Err_ModelNull;

    QModelIndex idx = pModel->nodeIndex(hNode);
    if (!idx.isValid())
        return Err_NodeNotExist;

    INPUT_SOCKETS inputs = idx.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
    const QString& qsInSock = QString::fromStdString(inSock);
    if (inputs.find(qsInSock) == inputs.end())
        return Err_SockNotExist;

    const QString& nodeid = idx.data(ROLE_OBJID).toString();
    INPUT_SOCKET input = inputs[qsInSock];
    ret = ApiUtil::qVarToStdVar(input.info.defaultValue);
    type = input.info.type.toStdString();
    return Err_NoError;
}

ZENO_ERROR Zeno_SetInputDefl(
        ZENO_HANDLE hNode,
        const std::string& inSock,
        const ZVARIANT& var)
{
    IGraphsModel *pModel = GraphsManagment::instance().currentModel();
    if (!pModel)
        return Err_ModelNull;

    QModelIndex idx = pModel->nodeIndex(hNode);
    if (!idx.isValid())
        return Err_NodeNotExist;

    INPUT_SOCKETS inputs = idx.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
    const QString &qsInSock = QString::fromStdString(inSock);
    if (inputs.find(qsInSock) == inputs.end())
        return Err_SockNotExist;

    const QString& nodeid = idx.data(ROLE_OBJID).toString();
    const QModelIndex& subgIdx = pModel->subgByNodeId(hNode);
    INPUT_SOCKET input = inputs[qsInSock];

    PARAM_UPDATE_INFO info;
    info.name = qsInSock;
    info.newValue = ApiUtil::stdVarToQVar(var);
    info.oldValue = input.info.defaultValue;
    pModel->updateSocketDefl(nodeid, info, subgIdx);
    return Err_NoError;
}

ZENO_ERROR Zeno_GetParam(
        ZENO_HANDLE hNode,
        const std::string& name,
        /*out*/ ZVARIANT& ret,
        /*out*/ std::string& type)
{
    IGraphsModel *pModel = GraphsManagment::instance().currentModel();
    if (!pModel)
        return Err_ModelNull;

    QModelIndex idx = pModel->nodeIndex(hNode);
    if (!idx.isValid())
        return Err_NodeNotExist;

    QString qsName = QString::fromStdString(name);
    PARAMS_INFO params = idx.data(ROLE_PARAMETERS).value<PARAMS_INFO>();
    if (params.find(qsName) == params.end())
    {
        return Err_ParamNotFound;
    }

    PARAM_INFO param = params[qsName];
    ret = ApiUtil::qVarToStdVar(param.value);
    type = param.typeDesc.toStdString();
    return Err_NoError;
}

ZENO_ERROR Zeno_SetParam(
        ZENO_HANDLE hNode,
        const std::string& name,
        const ZVARIANT& var)
{
    IGraphsModel *pModel = GraphsManagment::instance().currentModel();
    if (!pModel)
        return Err_ModelNull;

    const QModelIndex& idx = pModel->nodeIndex(hNode);
    QString qsName = QString::fromStdString(name);
    PARAMS_INFO params = idx.data(ROLE_PARAMETERS).value<PARAMS_INFO>();
    if (params.find(qsName) == params.end())
    {
        return Err_ParamNotFound;
    }

    PARAM_INFO param = params[qsName];
    const QModelIndex& subgIdx = pModel->subgByNodeId(hNode);

    PARAM_UPDATE_INFO info;
    info.name = qsName;
    info.newValue = ApiUtil::stdVarToQVar(var);
    info.oldValue = param.value;

    pModel->updateParamInfo(idx.data(ROLE_OBJID).toString(), info, subgIdx);
    return Err_NoError;
}

ZENO_ERROR Zeno_IsView(ZENO_HANDLE hNode, bool& ret)
{
    IGraphsModel *pModel = GraphsManagment::instance().currentModel();
    if (!pModel)
        return Err_ModelNull;

    QModelIndex idx = pModel->nodeIndex(hNode);
    if (!idx.isValid())
        return Err_NodeNotExist;

    int opts = idx.data(ROLE_OPTIONS).toInt();
    ret = (opts & OPT_VIEW);
    return Err_NoError;
}

ZENO_ERROR Zeno_SetView(ZENO_HANDLE hNode, bool bOn)
{
    IGraphsModel *pModel = GraphsManagment::instance().currentModel();
    if (!pModel)
        return Err_ModelNull;

    QModelIndex idx = pModel->nodeIndex(hNode);
    if (!idx.isValid())
        return Err_NodeNotExist;

    QModelIndex subgIdx = pModel->subgByNodeId(hNode);

    STATUS_UPDATE_INFO info;
    int options = idx.data(ROLE_OPTIONS).toInt();
    info.oldValue = options;
    if (bOn) {
        options |= OPT_VIEW;
    } else {
        options ^= OPT_VIEW;
    }
    info.role = ROLE_OPTIONS;
    info.newValue = options;

    pModel->updateNodeStatus(idx.data(ROLE_OBJID).toString(), info, subgIdx);
    return Err_NoError;
}

ZENO_ERROR Zeno_IsMute(ZENO_HANDLE hNode, bool& ret)
{
    IGraphsModel *pModel = GraphsManagment::instance().currentModel();
    if (!pModel)
        return Err_ModelNull;

    QModelIndex idx = pModel->nodeIndex(hNode);
    if (!idx.isValid())
        return Err_NodeNotExist;

    int opts = idx.data(ROLE_OPTIONS).toInt();
    ret = (opts & OPT_MUTE);
    return Err_NoError;
}

ZENO_ERROR Zeno_SetMute(ZENO_HANDLE hNode, bool bOn)
{
    IGraphsModel *pModel = GraphsManagment::instance().currentModel();
    if (!pModel)
        return Err_ModelNull;

    QModelIndex idx = pModel->nodeIndex(hNode);
    if (!idx.isValid())
        return Err_NodeNotExist;

    QModelIndex subgIdx = pModel->subgByNodeId(hNode);

    STATUS_UPDATE_INFO info;
    int options = idx.data(ROLE_OPTIONS).toInt();
    info.oldValue = options;
    if (bOn) {
        options |= OPT_MUTE;
    } else {
        options ^= OPT_MUTE;
    }
    info.role = ROLE_OPTIONS;
    info.newValue = options;

    pModel->updateNodeStatus(idx.data(ROLE_OBJID).toString(), info, subgIdx);
    return Err_NoError;
}

ZENO_ERROR Zeno_IsOnce(ZENO_HANDLE hNode, bool& ret)
{
    IGraphsModel *pModel = GraphsManagment::instance().currentModel();
    if (!pModel)
        return Err_ModelNull;

    QModelIndex idx = pModel->nodeIndex(hNode);
    if (!idx.isValid())
        return Err_NodeNotExist;

    int opts = idx.data(ROLE_OPTIONS).toInt();
    ret = (opts & OPT_ONCE);
    return Err_NoError;
}

ZENO_ERROR Zeno_SetOnce(ZENO_HANDLE hNode, bool bOn)
{
    IGraphsModel *pModel = GraphsManagment::instance().currentModel();
    if (!pModel)
        return Err_ModelNull;

    QModelIndex idx = pModel->nodeIndex(hNode);
    if (!idx.isValid())
        return Err_NodeNotExist;

    QModelIndex subgIdx = pModel->subgByNodeId(hNode);

    STATUS_UPDATE_INFO info;
    int options = idx.data(ROLE_OPTIONS).toInt();
    info.oldValue = options;
    if (bOn) {
        options |= OPT_ONCE;
    } else {
        options ^= OPT_ONCE;
    }
    info.role = ROLE_OPTIONS;
    info.newValue = options;

    pModel->updateNodeStatus(idx.data(ROLE_OBJID).toString(), info, subgIdx);
    return Err_NoError;
}

ZENO_ERROR Zeno_GetPos(ZENO_HANDLE hNode, std::pair<float, float>& pt)
{
    IGraphsModel* pModel = GraphsManagment::instance().currentModel();
    if (!pModel)
        return Err_ModelNull;

    QModelIndex idx = pModel->nodeIndex(hNode);
    if (!idx.isValid())
        return Err_NodeNotExist;

    QPointF pos = idx.data(ROLE_OBJPOS).toPointF();
    pt.first = pos.x();
    pt.second = pos.y();
    return Err_NoError;
}

ZENO_ERROR Zeno_SetPos(ZENO_HANDLE hNode, const std::pair<float, float>& pt)
{
    IGraphsModel* pModel = GraphsManagment::instance().currentModel();
    if (!pModel)
        return Err_ModelNull;

    QModelIndex idx = pModel->nodeIndex(hNode);
    if (!idx.isValid())
        return Err_NodeNotExist;

    QModelIndex subgIdx = pModel->subgByNodeId(hNode);

    const QString& nodeid = idx.data(ROLE_OBJID).toString();
    QPointF oldPos = idx.data(ROLE_OBJPOS).toPointF();
    QPointF newPos = {pt.first, pt.second};

    STATUS_UPDATE_INFO info;
    info.newValue = newPos;
    info.oldValue = oldPos;
    info.role = ROLE_OBJPOS;

    pModel->updateNodeStatus(nodeid, info, subgIdx, false);
    return Err_NoError;
}
