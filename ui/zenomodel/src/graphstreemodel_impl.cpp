#include "graphstreemodel_impl.h"
#include "nodeitem.h"
#include "linkmodel.h"
#include "nodeparammodel.h"
#include "panelparammodel.h"
#include "modelrole.h"
#include "modeldata.h"
#include "variantptr.h"
#include "uihelper.h"
#include "graphstreemodel.h"
#include "command.h"
#include "apilevelscope.h"
#include "common_def.h"


GraphsTreeModel_impl::GraphsTreeModel_impl(GraphsTreeModel* pModel, QObject *parent)
    : QStandardItemModel(parent)
    , m_linkModel(nullptr)
    , m_pModel(pModel)
{
    NODE_DATA dat;
    dat[ROLE_OBJNAME] = "main";
    dat[ROLE_NODETYPE] = SUBGRAPH_NODE;
    dat[ROLE_OBJID] = "main";
    m_main = new TreeNodeItem(dat, m_pModel);
    m_linkModel = new LinkModel(this);
    appendRow(m_main);
}

GraphsTreeModel_impl::~GraphsTreeModel_impl()
{
}

QModelIndex GraphsTreeModel_impl::index(int row, int column, const QModelIndex& parent) const
{
    return QStandardItemModel::index(row, column, parent);
}

QModelIndex GraphsTreeModel_impl::index(const QString &subGraphName) const
{
    return QModelIndex();
}

QModelIndex GraphsTreeModel_impl::index(const QString &id, const QModelIndex &subGpIdx)
{
    TreeNodeItem* pSubgraphItem = static_cast<TreeNodeItem*>(this->itemFromIndex(subGpIdx));
    if (!pSubgraphItem)
        return QModelIndex();
    return pSubgraphItem->childIndex(id);
}

QModelIndex GraphsTreeModel_impl::index(int r, const QModelIndex &subGpIdx)
{
    TreeNodeItem *pSubgraphItem = static_cast<TreeNodeItem *>(this->itemFromIndex(subGpIdx));
    if (!pSubgraphItem)
        return QModelIndex();

    if (r < 0 || r >= pSubgraphItem->rowCount())
        return QModelIndex();

    return pSubgraphItem->child(r)->index();
}

QModelIndex GraphsTreeModel_impl::mainIndex() const
{
    return m_main->index();
}

QModelIndex GraphsTreeModel_impl::nodeIndex(const QString &ident)
{
    //todo: deprecated.
    return QModelIndex();
}

QModelIndex GraphsTreeModel_impl::nodeIndex(uint32_t sid, uint32_t nodeid)
{
    //todo: deprecated.
    return QModelIndex();
}

QModelIndex GraphsTreeModel_impl::subgIndex(uint32_t sid)
{
    //todo: deprecated.
    return QModelIndex();
}

void GraphsTreeModel_impl::initMainGraph()
{
}

void GraphsTreeModel_impl::clear()
{
    QStandardItemModel::clear();
    m_main = nullptr;
}

int GraphsTreeModel_impl::itemCount(const QModelIndex &subGpIdx) const
{
    return this->rowCount(subGpIdx);
}

QModelIndex GraphsTreeModel_impl::linkIndex(const QModelIndex &subgIdx, int r)
{
    ZASSERT_EXIT(m_linkModel, QModelIndex());
    return m_linkModel->index(r, 0);
}

QModelIndex GraphsTreeModel_impl::linkIndex(
                        const QModelIndex &subgIdx,
                        const QString &outNode,
                        const QString &outSock,
                        const QString &inNode,
                        const QString &inSock)
{
    //todo: deprecated
    return QModelIndex();
}

void GraphsTreeModel_impl::onSubIOAddRemove(
                        TreeNodeItem* pSubgraph,
                        const QModelIndex& addedNodeIdx,
                        bool bInput,
                        bool bInsert)
{
    //reject main function.
    if (pSubgraph->parent() == invisibleRootItem())
        return;

    ZASSERT_EXIT(pSubgraph);

    const QString& objId = addedNodeIdx.data(ROLE_OBJID).toString();
    const QString& objName = addedNodeIdx.data(ROLE_OBJNAME).toString();

    NodeParamModel* nodeParams = QVariantPtr<NodeParamModel>::asPtr(addedNodeIdx.data(ROLE_NODE_PARAMS));

    const QModelIndex& nameIdx = nodeParams->getParam(PARAM_PARAM, "name");
    const QModelIndex& typeIdx = nodeParams->getParam(PARAM_PARAM, "type");
    const QModelIndex& deflIdx = nodeParams->getParam(PARAM_PARAM, "defl");
    ZASSERT_EXIT(nameIdx.isValid() && typeIdx.isValid() && deflIdx.isValid());

    const QString& nameValue = nameIdx.data(ROLE_PARAM_VALUE).toString();
    const QString& typeValue = typeIdx.data(ROLE_PARAM_VALUE).toString();
    QVariant deflVal = deflIdx.data(ROLE_PARAM_VALUE);
    const PARAM_CONTROL ctrl = (PARAM_CONTROL)deflIdx.data(ROLE_PARAM_CTRL).toInt();
    QVariant ctrlProps = deflIdx.data(ROLE_VPARAM_CTRL_PROPERTIES);
    QString toolTip = nameIdx.data(ROLE_VPARAM_TOOLTIP).toString();

    //only need to update pSubgraph node.
    if (bInsert)
    {
        nodeParams->setAddParam(
                        bInput ? PARAM_INPUT : PARAM_OUTPUT,
                        nameValue,
                        typeValue,
                        deflVal,
                        ctrl,
                        ctrlProps,
                        SOCKPROP_NORMAL,
                        DICTPANEL_INFO(),
                        toolTip);
    }
    else
    {
        nodeParams->removeParam(bInput ? PARAM_INPUT : PARAM_OUTPUT, nameValue);
    }
}

bool GraphsTreeModel_impl::onSubIOAdd(TreeNodeItem *pSubgraph, NODE_DATA nodeData)
{
    const QString& descName = nodeData[ROLE_OBJNAME].toString();
    if (descName != "SubInput" && descName != "SubOutput")
        return false;

    bool bInput = descName == "SubInput";

    PARAMS_INFO params = nodeData[ROLE_PARAMETERS].value<PARAMS_INFO>();
    ZASSERT_EXIT(params.find("name") != params.end(), false);
    PARAM_INFO& param = params["name"];
    QString newSockName = UiHelper::correctSubIOName(m_pModel, pSubgraph->objClass(), param.value.toString(), bInput);
    param.value = newSockName;
    nodeData[ROLE_PARAMETERS] = QVariant::fromValue(params);

    pSubgraph->addNode(nodeData, m_pModel);

    if (!m_pModel->IsIOProcessing()) {
        const QString& ident = nodeData[ROLE_OBJID].toString();
        const QModelIndex& nodeIdx = pSubgraph->childIndex(ident);
        onSubIOAddRemove(pSubgraph, nodeIdx, bInput, true);
    }
    return true;
}

bool GraphsTreeModel_impl::onListDictAdd(TreeNodeItem* pSubgraph, NODE_DATA nodeData)
{
    const QString& descName = nodeData[ROLE_OBJNAME].toString();
    if (descName == "MakeList" || descName == "MakeDict")
    {
        INPUT_SOCKETS inputs = nodeData[ROLE_INPUTS].value<INPUT_SOCKETS>();
        INPUT_SOCKET inSocket;
        inSocket.info.nodeid = nodeData[ROLE_OBJID].toString();

        int maxObjId = UiHelper::getMaxObjId(inputs.keys());
        if (maxObjId == -1)
        {
            inSocket.info.name = "obj0";
            if (descName == "MakeDict") {
                inSocket.info.control = CONTROL_NONE;
                inSocket.info.sockProp = SOCKPROP_EDITABLE;
            }
            inputs.insert(inSocket.info.name, inSocket);
            nodeData[ROLE_INPUTS] = QVariant::fromValue(inputs);
        }
        pSubgraph->addNode(nodeData, m_pModel);
        return true;
    }
    else if (descName == "ExtractDict")
    {
        OUTPUT_SOCKETS outputs = nodeData[ROLE_OUTPUTS].value<OUTPUT_SOCKETS>();
        OUTPUT_SOCKET outSocket;
        outSocket.info.nodeid = nodeData[ROLE_OBJID].toString();

        int maxObjId = UiHelper::getMaxObjId(outputs.keys());
        if (maxObjId == -1) {
            outSocket.info.name = "obj0";
            outSocket.info.control = CONTROL_NONE;
            outSocket.info.sockProp = SOCKPROP_EDITABLE;
            outputs.insert(outSocket.info.name, outSocket);
            nodeData[ROLE_OUTPUTS] = QVariant::fromValue(outputs);
        }
        pSubgraph->addNode(nodeData, m_pModel);
        return true;
    }
    return false;
}

void GraphsTreeModel_impl::addNode(const NODE_DATA& nodeData, const QModelIndex& subGpIdx, bool enableTransaction)
{
    bool bEnableIOProc = m_pModel->IsIOProcessing();
    if (bEnableIOProc)
        enableTransaction = false;

    if (enableTransaction)
    {
        QString id = nodeData[ROLE_OBJID].toString();
        AddNodeCommand *pCmd = new AddNodeCommand(id, nodeData, m_pModel, subGpIdx);
        m_pModel->stack()->push(pCmd);
    }
    else
    {
        ApiLevelScope batch(m_pModel);

        TreeNodeItem* pSubgItem = static_cast<TreeNodeItem*>(itemFromIndex(subGpIdx));
        ZASSERT_EXIT(pSubgItem);

        if (onSubIOAdd(pSubgItem, nodeData))
            return;
        if (onListDictAdd(pSubgItem, nodeData))
            return;
        pSubgItem->addNode(nodeData, m_pModel);
    }
}

QList<EdgeInfo> GraphsTreeModel_impl::addSubnetNode(
            IGraphsModel* pSubgraphs,
            const QString& subnetName,
            const QString& ident,
            const QString& customName)
{
    QList<EdgeInfo> newLinks;

    QModelIndex idxSharedSubg = pSubgraphs->index(subnetName);
    ZASSERT_EXIT(idxSharedSubg.isValid(), newLinks);

    NODE_DATA nodeData;
    nodeData[ROLE_OBJID] = ident;
    nodeData[ROLE_OBJNAME] = subnetName;
    nodeData[ROLE_CUSTOM_OBJNAME] = customName;
    nodeData[ROLE_COLLASPED] = false;
    nodeData[ROLE_NODETYPE] = SUBGRAPH_NODE;

    TreeNodeItem *pNewSubnetItem = _fork("/main/" + ident, pSubgraphs, subnetName, nodeData, newLinks);
    ZASSERT_EXIT(pNewSubnetItem, newLinks);
    m_main->appendRow(pNewSubnetItem);
    return newLinks;
}

TreeNodeItem* GraphsTreeModel_impl::_fork(
                const QString& currentPath,
                IGraphsModel* pSubgraphs,
                const QString& subnetName,
                const NODE_DATA& nodeData,
                QList<EdgeInfo>& newLinks)
{
    TreeNodeItem* pSubnetNode = new TreeNodeItem(nodeData, m_pModel);

    QMap<QString, NODE_DATA> nodes;
    QMap<QString, TreeNodeItem*> oldGraphsToNew;
    QHash<QString, QString> old2new;
    QHash<QString, QString> old2new_nodePath;
    QVector<EdgeInfo> oldLinks;
    
    QModelIndex sharedSubg = pSubgraphs->index(subnetName);
    ZASSERT_EXIT(sharedSubg.isValid(), nullptr);
    for (int r = 0; r < pSubgraphs->itemCount(sharedSubg); r++)
    {
        QModelIndex nodeIdx = pSubgraphs->index(r, sharedSubg);
        NODE_DATA nodeData = nodeIdx.data(ROLE_OBJDATA).value<NODE_DATA>();
        const QString &snodeId = nodeIdx.data(ROLE_OBJID).toString();
        const QString &name = nodeData[ROLE_OBJNAME].toString();
        const QString &newId = UiHelper::generateUuid(name);
        old2new.insert(snodeId, newId);

        TreeNodeItem* newNodeItem = nullptr;
        if (pSubgraphs->IsSubGraphNode(nodeIdx))
        {
            const QString &ssubnetName = nodeIdx.data(ROLE_OBJNAME).toString();
            nodeData[ROLE_OBJID] = newId;
            nodeData[ROLE_NODETYPE] = SUBGRAPH_NODE;
            newNodeItem = _fork(currentPath + "/" + newId, pSubgraphs, ssubnetName, nodeData, newLinks);
            nodes.insert(snodeId, nodeData);
        }
        else
        {
            nodeData[ROLE_OBJID] = newId;
            newNodeItem = new TreeNodeItem(nodeData, m_pModel);
        }
        pSubnetNode->appendRow(newNodeItem);

        //apply legacy format `subnet:nodeid`.
        const QString &oldNodePath = QString("%1:%2").arg(subnetName).arg(snodeId);
        const QString &newNodePath = currentPath + "/" + newId;
        old2new_nodePath.insert(oldNodePath, newNodePath);

        //only collect links from input socket.
        NodeParamModel *viewParams = QVariantPtr<NodeParamModel>::asPtr(nodeIdx.data(ROLE_NODE_PARAMS));
        const QModelIndexList &lst = viewParams->getInputIndice();
        for (int r = 0; r < lst.size(); r++)
        {
            const QModelIndex &paramIdx = lst[r];
            const QString &inSock = paramIdx.data(ROLE_PARAM_NAME).toString();
            const int inSockProp = paramIdx.data(ROLE_PARAM_SOCKPROP).toInt();
            PARAM_LINKS links = paramIdx.data(ROLE_PARAM_LINKS).value<PARAM_LINKS>();
            if (!links.isEmpty())
            {
                for (auto linkIdx : links)
                {
                    oldLinks.append(UiHelper::exportLink(linkIdx));
                }
            }
            else if (inSockProp & SOCKPROP_DICTLIST_PANEL)
            {
                QAbstractItemModel *pKeyObjModel =
                    QVariantPtr<QAbstractItemModel>::asPtr(paramIdx.data(ROLE_VPARAM_LINK_MODEL));
                for (int _r = 0; _r < pKeyObjModel->rowCount(); _r++)
                {
                    const QModelIndex &keyIdx = pKeyObjModel->index(_r, 0);
                    ZASSERT_EXIT(keyIdx.isValid(), nullptr);
                    PARAM_LINKS links = keyIdx.data(ROLE_PARAM_LINKS).value<PARAM_LINKS>();
                    if (!links.isEmpty())
                    {
                        const QModelIndex &linkIdx = links[0];
                        oldLinks.append(UiHelper::exportLink(linkIdx));
                    }
                }
            }
        }
    }

    for (EdgeInfo oldLink : oldLinks)
    {
        //oldLink format: subg:xxx-objid:sockid
        QString outputNode = UiHelper::getNodePath(oldLink.outSockPath);
        QString outParamPath = UiHelper::getParamPath(oldLink.outSockPath);
        QString inputNode = UiHelper::getNodePath(oldLink.inSockPath);
        QString inParamPath = UiHelper::getParamPath(oldLink.inSockPath);

        ZASSERT_EXIT(old2new_nodePath.find(outputNode) != old2new_nodePath.end() &&
                     old2new_nodePath.find(inputNode) != old2new_nodePath.end(), nullptr);

        const QString& newInSock = old2new_nodePath[inputNode] + cPathSeperator + inParamPath;
        const QString& newOutSock = old2new_nodePath[outputNode] + cPathSeperator + outParamPath;
        newLinks.append(EdgeInfo(newOutSock, newInSock));
    }

    return pSubnetNode;
}


void GraphsTreeModel_impl::setNodeData(const QModelIndex &nodeIndex, const QModelIndex &subGpIdx, const QVariant &value, int role)
{
    //todo: deprecated.
}

void GraphsTreeModel_impl::importNodes(
                    const QMap<QString, NODE_DATA> &nodes,
                    const QList<EdgeInfo> &links,
                    const QPointF &pos,
                    const QModelIndex &subGpIdx,
                    bool enableTransaction)
{
    if (nodes.isEmpty())
        return;

    if (enableTransaction)
    {
        ImportNodesCommand *pCmd = new ImportNodesCommand(nodes, links, pos, m_pModel, subGpIdx);
        m_pModel->stack()->push(pCmd);
    }
    else
    {
        ApiLevelScope batch(m_pModel);

        TreeNodeItem *pSubgItem = static_cast<TreeNodeItem *>(itemFromIndex(subGpIdx));
        ZASSERT_EXIT(pSubgItem);

        for (const NODE_DATA &data : nodes)
        {
            addNode(data, subGpIdx, false);
        }

        //resolve pos and links.
        QStringList ids = nodes.keys();
        QModelIndex nodeIdx = index(ids[0], subGpIdx);
        QPointF _pos = nodeIdx.data(ROLE_OBJPOS).toPointF();
        const QPointF offset = pos - _pos;

        for (const QString &ident : ids)
        {
            const QModelIndex &idx = pSubgItem->childIndex(ident);
            _pos = idx.data(ROLE_OBJPOS).toPointF();
            _pos += offset;
            setData(idx, _pos, ROLE_OBJPOS);
        }
        for (EdgeInfo link : links)
        {
            addLink(link, false);
        }
    }
}

void GraphsTreeModel_impl::removeNode(
                        const QString& nodeid,
                        const QModelIndex& subGpIdx,
                        bool enableTransaction)
{
    bool bEnableIOProc = m_pModel->IsIOProcessing();
    if (bEnableIOProc)
        enableTransaction = false;

    TreeNodeItem *pSubgItem = static_cast<TreeNodeItem *>(itemFromIndex(subGpIdx));
    ZASSERT_EXIT(pSubgItem);

    if (enableTransaction)
    {
        int row = pSubgItem->id2Row(nodeid);
        ZASSERT_EXIT(row != -1);
        TreeNodeItem* pNodeItem = pSubgItem->childItem(nodeid);
        ZASSERT_EXIT(pNodeItem);
        const NODE_DATA& dat = pNodeItem->expData();

        RemoveNodeCommand *pCmd = new RemoveNodeCommand(row, dat, m_pModel, subGpIdx);
        m_pModel->stack()->push(pCmd);
    }
    else
    {
        ApiLevelScope batch(m_pModel);

        QModelIndex idx = pSubgItem->childIndex(nodeid);
        const QString &objName = idx.data(ROLE_OBJNAME).toString();
        if (!bEnableIOProc)
        {
            //if subnode removed, the parent layer node should be update.
            if (objName == "SubInput") {
                onSubIOAddRemove(pSubgItem, idx, true, false);
            } else if (objName == "SubOutput") {
                onSubIOAddRemove(pSubgItem, idx, false, false);
            }
        }
        pSubgItem->removeNode(nodeid, m_pModel);
    }
}


QModelIndex GraphsTreeModel_impl::addLink(
                        const QModelIndex& subgIdx,
                        const QModelIndex& fromSock,
                        const QModelIndex& toSock,
                        bool enableTransaction)
{
    ZASSERT_EXIT(fromSock.isValid() && toSock.isValid(), QModelIndex());
    EdgeInfo link(fromSock.data(ROLE_OBJPATH).toString(), toSock.data(ROLE_OBJPATH).toString());
    return addLink(link, enableTransaction);
}

QModelIndex GraphsTreeModel_impl::addLink(
                        const EdgeInfo& info,
                        bool enableTransaction)
{
    if (enableTransaction)
    {
        LinkCommand *pCmd = new LinkCommand(QModelIndex(), true, info, m_pModel);
        m_pModel->stack()->push(pCmd);
        //todo: return val on this level.
        return QModelIndex();
    }
    else
    {
        ApiLevelScope batch(m_pModel);

        QModelIndex inParamIdx = indexFromPath(info.inSockPath);
        QModelIndex outParamIdx = indexFromPath(info.outSockPath);
        if (!inParamIdx.isValid() || !outParamIdx.isValid()) {
            zeno::log_warn("there is not valid input or output sockets.");
            return QModelIndex();
        }

        ZASSERT_EXIT(m_linkModel, QModelIndex());

        int row = m_linkModel->addLink(outParamIdx, inParamIdx);
        const QModelIndex& linkIdx = m_linkModel->index(row, 0);

        QAbstractItemModel* pInputs = const_cast<QAbstractItemModel*>(inParamIdx.model());
        QAbstractItemModel* pOutputs = const_cast<QAbstractItemModel*>(outParamIdx.model());

        ZASSERT_EXIT(pInputs && pOutputs, QModelIndex());
        pInputs->setData(inParamIdx, linkIdx, ROLE_ADDLINK);
        pOutputs->setData(outParamIdx, linkIdx, ROLE_ADDLINK);
        return linkIdx;
    }
}

void GraphsTreeModel_impl::removeLink(
                        const QModelIndex& linkIdx,
                        bool enableTransaction)
{
    if (!linkIdx.isValid())
        return;

    QModelIndex inSockIdx = linkIdx.data(ROLE_INSOCK_IDX).toModelIndex();
    QModelIndex outSockIdx = linkIdx.data(ROLE_OUTSOCK_IDX).toModelIndex();
    QModelIndex nodeIdx = outSockIdx.data(ROLE_NODE_IDX).toModelIndex();
    QModelIndex subgIdx = nodeIdx.data(ROLE_SUBGRAPH_IDX).toModelIndex();

    ZASSERT_EXIT(inSockIdx.isValid() && outSockIdx.isValid());
    EdgeInfo link(outSockIdx.data(ROLE_OBJPATH).toString(), inSockIdx.data(ROLE_OBJPATH).toString());
    removeLink(subgIdx, link, enableTransaction);
}

void GraphsTreeModel_impl::removeLink(
                        const QModelIndex& subgIdx,
                        const EdgeInfo& link,
                        bool enableTransaction)
{
    if (enableTransaction)
    {
        LinkCommand *pCmd = new LinkCommand(subgIdx, false, link, m_pModel);
        m_pModel->stack()->push(pCmd);
    }
    else
    {
        ApiLevelScope batch(m_pModel);

        //sometimes when removing socket, the link attached on it will also be removed,
        //but if the socket index is invalid, then cause the associated link cannot be restored by these sockets.
        //so, we must ensure the removal of link, is ahead of the removal of sockets.

        //find the socket idx
        const QModelIndex &outSockIdx = indexFromPath(link.outSockPath);
        const QModelIndex &inSockIdx = indexFromPath(link.inSockPath);
        ZASSERT_EXIT(outSockIdx.isValid() && inSockIdx.isValid());

        LinkModel *pLinkModel = linkModel(subgIdx);
        ZASSERT_EXIT(pLinkModel);

        //restore the link
        QModelIndex linkIdx = pLinkModel->index(outSockIdx, inSockIdx);

        QAbstractItemModel *pOutputs = const_cast<QAbstractItemModel *>(outSockIdx.model());
        ZASSERT_EXIT(pOutputs);
        pOutputs->setData(outSockIdx, linkIdx, ROLE_REMOVELINK);

        QAbstractItemModel *pInputs = const_cast<QAbstractItemModel *>(inSockIdx.model());
        ZASSERT_EXIT(pInputs);
        pInputs->setData(inSockIdx, linkIdx, ROLE_REMOVELINK);

        ZASSERT_EXIT(linkIdx.isValid());
        pLinkModel->removeRow(linkIdx.row());
    }
}

bool GraphsTreeModel_impl::IsSubGraphNode(const QModelIndex& nodeIdx) const
{
    return nodeIdx.data(ROLE_NODETYPE).toInt() == SUBGRAPH_NODE;
}

QModelIndex GraphsTreeModel_impl::fork(const QModelIndex &subgIdx, const QModelIndex &subnetNodeIdx)
{
    //todo: deprecated in this implementation.
    return QModelIndex();
}

void GraphsTreeModel_impl::updateParamInfo(
                    const QString& id,
                    PARAM_UPDATE_INFO info,
                    const QModelIndex& subGpIdx,
                    bool enableTransaction)
{
    const QModelIndex& nodeIdx = index(id, subGpIdx);
    NodeParamModel* nodeParams = QVariantPtr<NodeParamModel>::asPtr(nodeIdx.data(ROLE_NODE_PARAMS));
    const QModelIndex& paramIdx = nodeParams->getParam(PARAM_PARAM, info.name);
    ModelSetData(paramIdx, info.newValue, ROLE_PARAM_VALUE);
}

void GraphsTreeModel_impl::updateSocketDefl(
                    const QString& id,
                    PARAM_UPDATE_INFO info,
                    const QModelIndex& subGpIdx,
                    bool enableTransaction)
{
    const QModelIndex& nodeIdx = index(id, subGpIdx);
    NodeParamModel* nodeParams = QVariantPtr<NodeParamModel>::asPtr(nodeIdx.data(ROLE_NODE_PARAMS));
    const QModelIndex& paramIdx = nodeParams->getParam(PARAM_INPUT, info.name);
    ModelSetData(paramIdx, info.newValue, ROLE_PARAM_VALUE);
}

void GraphsTreeModel_impl::updateNodeStatus(
                    const QString& nodeid,
                    STATUS_UPDATE_INFO info,
                    const QModelIndex& subgIdx,
                    bool enableTransaction)
{
    QModelIndex nodeIdx = index(nodeid, subgIdx);
    ModelSetData(nodeIdx, info.newValue, info.role);
}

void GraphsTreeModel_impl::updateBlackboard(
                    const QString& id,
                    const QVariant& newInfo,
                    const QModelIndex& subgIdx,
                    bool enableTransaction)
{
    TreeNodeItem* pSubgItem = static_cast<TreeNodeItem*>(itemFromIndex(subgIdx));
    ZASSERT_EXIT(pSubgItem);

    const QModelIndex& idx = pSubgItem->childIndex(id);

    if (enableTransaction)
    {
        if (newInfo.canConvert<BLACKBOARD_INFO>())
        {
            PARAMS_INFO params = idx.data(ROLE_PARAMS_NO_DESC).value<PARAMS_INFO>();
            BLACKBOARD_INFO oldInfo = params["blackboard"].value.value<BLACKBOARD_INFO>();
            UpdateBlackboardCommand *pCmd =
                new UpdateBlackboardCommand(id, newInfo.value<BLACKBOARD_INFO>(), oldInfo, m_pModel, subgIdx);
            m_pModel->stack()->push(pCmd);
        }
        else if (newInfo.canConvert<STATUS_UPDATE_INFO>())
        {
            updateNodeStatus(id, newInfo.value<STATUS_UPDATE_INFO>(), subgIdx, enableTransaction);
        }
    }
    else
    {
        if (newInfo.canConvert<BLACKBOARD_INFO>())
        {
            PARAMS_INFO params = idx.data(ROLE_PARAMS_NO_DESC).value<PARAMS_INFO>();
            params["blackboard"].name = "blackboard";
            params["blackboard"].value = QVariant::fromValue(newInfo);
            setData(idx, QVariant::fromValue(params), ROLE_PARAMS_NO_DESC);
        }
        else if (newInfo.canConvert<STATUS_UPDATE_INFO>())
        {
            setData(idx, newInfo.value<STATUS_UPDATE_INFO>().newValue, ROLE_OBJPOS);
        }
    }
}

NODE_DATA GraphsTreeModel_impl::itemData(const QModelIndex& index, const QModelIndex& subGpIdx) const
{
    TreeNodeItem* pSubgItem = static_cast<TreeNodeItem*>(itemFromIndex(subGpIdx));
    ZASSERT_EXIT(pSubgItem, NODE_DATA());
    TreeNodeItem* pChildItem = static_cast<TreeNodeItem*>(pSubgItem->child(index.row()));
    ZASSERT_EXIT(pChildItem, NODE_DATA());
    return pChildItem->expData();
}

int GraphsTreeModel_impl::ModelSetData(
                    const QPersistentModelIndex& idx,
                    const QVariant& value,
                    int role,
                    const QString& comment)
{
    if (!idx.isValid())
        return -1;

    QAbstractItemModel* pTargetModel = const_cast<QAbstractItemModel*>(idx.model());
    if (!pTargetModel)
        return -1;

    const QVariant &oldValue = pTargetModel->data(idx, role);
    if (oldValue == value)
        return -1;

    ModelDataCommand *pCmd = new ModelDataCommand(m_pModel, idx, oldValue, value, role);
    m_pModel->stack()->push(pCmd); //will call model->setData method.
    return 0;
}

QModelIndex GraphsTreeModel_impl::indexFromPath(const QString& path)
{
    //format example:  /main/xxxx-subgA/yyyy-subgB/xxx-Wrangle:[node]/outputs/prim
    //example2:        /main/xxxx-subgA/yyyy-subgB/xxx-Wrangle:[node]/inputs/dict/obj0
    //example3:        /main/xxxx-subgA/yyyy-subgB/xxx-Wrangle:[panel]/custom-group/custom-param1
    QStringList lst = path.split(cPathSeperator, QtSkipEmptyParts);
    if (lst.size() == 1)
    {
        const QString& nodePath = lst[0];
        lst = nodePath.split('/', QtSkipEmptyParts);

        TreeNodeItem *node = m_main, *p = m_main, *target = nullptr;
        if (lst[0] != "main")
            return QModelIndex();

        for (int i = 1; i < lst.size(); i++)
        {
            p = p->childItem(lst[i]);
            if (!p)
                break;
        }
        if (p)
            return p->index();
    }
    else if (lst.size() == 2)
    {
        const QString &nodePath = lst[0];
        QModelIndex nodeIdx = indexFromPath(nodePath);
        TreeNodeItem* pItem = static_cast<TreeNodeItem*>(itemFromIndex(nodeIdx));
        if (pItem == nullptr)
            return QModelIndex();

        const QString& paramPath = lst[1];
        if (paramPath.startsWith("[node]"))
        {
            const QString &paramObj = paramPath.mid(QString("[node]").length());
            ViewParamModel *viewParams = QVariantPtr<ViewParamModel>::asPtr(nodeIdx.data(ROLE_NODE_PARAMS));
            QModelIndex paramIdx = viewParams->indexFromPath(paramObj);
            return paramIdx;
        }
        else if (paramPath.startsWith("[panel]"))
        {
            const QString &paramObj = paramPath.mid(QString("[panel]").length());
            ViewParamModel *viewParams = QVariantPtr<ViewParamModel>::asPtr(nodeIdx.data(ROLE_PANEL_PARAMS));
            QModelIndex paramIdx = viewParams->indexFromPath(paramPath);
            return paramIdx;
        }
    }
    else if (lst.size() == 3)
    {
        //legacy case:    main:xxx-wrangle:/inputs/prim
        QString subnetName = lst[0];
        QString nodeid = lst[1];
        if (subnetName == "main")
        {
            QString newPath = QString("/main/%1:%2").arg(nodeid).arg(lst[2]);
            return indexFromPath(newPath);
        }
    }
    return QModelIndex();
}

void GraphsTreeModel_impl::setName(const QString &name, const QModelIndex &subGpIdx)
{
    //this name is obj class, which is not recommented to change name besides init io.
    this->setData(subGpIdx, name, ROLE_OBJNAME);
}

bool GraphsTreeModel_impl::setCustomName(const QModelIndex& subgIdx, const QModelIndex& idx, const QString& value)
{
    return setData(idx, value, ROLE_CUSTOM_OBJNAME);
}

QModelIndexList GraphsTreeModel_impl::searchInSubgraph(const QString& objName, const QModelIndex& idx)
{
    //todo:
    QModelIndexList list;
    return list;
}

QList<SEARCH_RESULT> GraphsTreeModel_impl::search(
                            const QString& content,
                            int searchType,
                            int searchOpts,
                            QVector<SubGraphModel*> vec)
{
    //todo:
    return QList<SEARCH_RESULT>();
}

QRectF GraphsTreeModel_impl::viewRect(const QModelIndex &subgIdx)
{
    return QRectF();
}

void GraphsTreeModel_impl::collaspe(const QModelIndex &subgIdx)
{
    TreeNodeItem* pSubgItem = static_cast<TreeNodeItem*>(itemFromIndex(subgIdx));
    ZASSERT_EXIT(pSubgItem);
    for (int i = 0; i < pSubgItem->rowCount(); i++)
    {
        TreeNodeItem* pChildItem = static_cast<TreeNodeItem*>(pSubgItem->child(i));
        pChildItem->setData(true, ROLE_COLLASPED);
    }
}

void GraphsTreeModel_impl::expand(const QModelIndex &subgIdx)
{
    TreeNodeItem* pSubgItem = static_cast<TreeNodeItem*>(itemFromIndex(subgIdx));
    ZASSERT_EXIT(pSubgItem);
    for (int i = 0; i < pSubgItem->rowCount(); i++)
    {
        TreeNodeItem* pChildItem = static_cast<TreeNodeItem*>(pSubgItem->child(i));
        pChildItem->setData(false, ROLE_COLLASPED);
    }
}

LinkModel* GraphsTreeModel_impl::linkModel(const QModelIndex &subgIdx) const
{
    return m_linkModel;
}