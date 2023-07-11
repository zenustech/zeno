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
#include <zenomodel/include/nodesmgr.h>
#include <zenoedit/zenoapplication.h>
#include "include/graphsmanagment.h"
#include "graphsmodel.h"


GraphsTreeModel_impl::GraphsTreeModel_impl(GraphsTreeModel* pModel, QObject *parent)
    : QStandardItemModel(parent)
    , m_linkModel(nullptr)
    , m_pModel(pModel)
{
    NODE_DATA dat;
    dat.nodeCls = "main";
    dat.type = SUBGRAPH_NODE;
    dat.ident = "main";
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

    nodeParams = QVariantPtr<NodeParamModel>::asPtr(pSubgraph->data(ROLE_NODE_PARAMS));
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
    //todo: SubInput socket init on `main` graph.
    if (m_pModel->IsIOProcessing())
        return false;

    const QString& descName = nodeData.nodeCls;
    if (descName != "SubInput" && descName != "SubOutput")
        return false;

    bool bInput = descName == "SubInput";

    ZASSERT_EXIT(nodeData.params.find("name") != nodeData.params.end(), false);
    PARAM_INFO& param = nodeData.params["name"];
    QString newSockName = UiHelper::correctSubIOName(pSubgraph->index(), param.value.toString(), bInput);
    param.value = newSockName;

    pSubgraph->addNode(nodeData, m_pModel);

    if (!m_pModel->IsIOProcessing()) {
        const QString& ident = nodeData.ident;
        const QModelIndex& nodeIdx = pSubgraph->childIndex(ident);
        onSubIOAddRemove(pSubgraph, nodeIdx, bInput, true);
    }
    return true;
}

bool GraphsTreeModel_impl::onListDictAdd(TreeNodeItem* pSubgraph, NODE_DATA nodeData)
{
    const QString& descName = nodeData.nodeCls;
    if (descName == "MakeList" || descName == "MakeDict")
    {
        INPUT_SOCKET inSocket;
        inSocket.info.nodeid = nodeData.ident;

        int maxObjId = UiHelper::getMaxObjId(nodeData.inputs.keys());
        if (maxObjId == -1)
        {
            inSocket.info.name = "obj0";
            if (descName == "MakeDict") {
                inSocket.info.control = CONTROL_NONE;
                inSocket.info.sockProp = SOCKPROP_EDITABLE;
            }
            nodeData.inputs.insert(inSocket.info.name, inSocket);
        }
        pSubgraph->addNode(nodeData, m_pModel);
        return true;
    }
    else if (descName == "ExtractDict")
    {
        OUTPUT_SOCKET outSocket;
        outSocket.info.nodeid = nodeData.ident;

        int maxObjId = UiHelper::getMaxObjId(nodeData.outputs.keys());
        if (maxObjId == -1) {
            outSocket.info.name = "obj0";
            outSocket.info.control = CONTROL_NONE;
            outSocket.info.sockProp = SOCKPROP_EDITABLE;
            nodeData.outputs.insert(outSocket.info.name, outSocket);
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
        QString id = nodeData.ident;
        AddNodeCommand *pCmd = new AddNodeCommand(id, nodeData, m_pModel, subGpIdx);
        m_pModel->stack()->push(pCmd);
    }
    else
    {
        ApiLevelScope batch(m_pModel);

        TreeNodeItem* pSubgItem = static_cast<TreeNodeItem*>(itemFromIndex(subGpIdx));
        ZASSERT_EXIT(pSubgItem);
        bool bAdd = false;
        bAdd = onSubIOAdd(pSubgItem, nodeData);
        if (!bAdd)
            bAdd = onListDictAdd(pSubgItem, nodeData);
        if (!bAdd)
            pSubgItem->addNode(nodeData, m_pModel);

        TreeNodeItem * childItem = pSubgItem->childItem(nodeData.ident);
        ZASSERT_EXIT(childItem);
        appendSubGraphNode(childItem);
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
    nodeData.ident = ident;
    nodeData.nodeCls = subnetName;
    nodeData.customName = customName;
    nodeData.bCollasped = false;
    nodeData.type = SUBGRAPH_NODE;

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
        const QString &name = nodeData.nodeCls;
        const QString &newId = UiHelper::generateUuid(name);
        old2new.insert(snodeId, newId);

        TreeNodeItem* newNodeItem = nullptr;
        if (pSubgraphs->IsSubGraphNode(nodeIdx))
        {
            const QString &ssubnetName = nodeIdx.data(ROLE_OBJNAME).toString();
            nodeData.ident = newId;
            nodeData.type = SUBGRAPH_NODE;
            newNodeItem = _fork(currentPath + "/" + newId, pSubgraphs, ssubnetName, nodeData, newLinks);
            nodes.insert(snodeId, nodeData);
        }
        else
        {
            nodeData.ident = newId;
            newNodeItem = new TreeNodeItem(nodeData, m_pModel);
        }
        pSubnetNode->appendRow(newNodeItem);

        //apply legacy format `subnet:nodeid`.
        const QString &oldNodePath = QString("%1/%2").arg(subnetName).arg(snodeId);
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
        removeSubGraphNode(pSubgItem->childItem(nodeid));
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
            zeno::log_error("there is not valid input or output sockets.");
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

void GraphsTreeModel_impl::exportSubgraph(const QModelIndex& subGpIdx, NODES_DATA& nodes, LINKS_DATA& links) const
{
    //no impl.
    TreeNodeItem* pSubgItem = static_cast<TreeNodeItem*>(itemFromIndex(subGpIdx));
    ZASSERT_EXIT(pSubgItem);
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

GraphsTreeModel* GraphsTreeModel_impl::model() const
{
    return m_pModel;
}

bool GraphsTreeModel_impl::setCustomName(const QModelIndex& subgIdx, const QModelIndex& idx, const QString& value)
{
    return setData(idx, value, ROLE_CUSTOM_OBJNAME);
}

QModelIndexList GraphsTreeModel_impl::searchInSubgraph(const QString& objName, const QModelIndex& subgIdx)
{
    QList<SEARCH_RESULT> results = search_impl(subgIdx, objName, SEARCHALL, SEARCH_FUZZ, false);
    QModelIndexList list;
    for (auto res : results) {
        list.append(res.targetIdx);
    }
    return list;
}

QList<SEARCH_RESULT> GraphsTreeModel_impl::search(
                            const QString& content,
                            int searchType,
                            int searchOpts)
{
    if (!m_main)
        return QList<SEARCH_RESULT>();
    const QModelIndex& mainIdx = m_main->index();
    return search_impl(mainIdx, content, searchType, searchOpts, true);
}

QList<SEARCH_RESULT> GraphsTreeModel_impl::search_impl(
            const QModelIndex& root,
            const QString &content,
            int searchType,
            int searchOpts,
            bool bRecursivly)
{
    QList<SEARCH_RESULT> results;
    if (content.isEmpty())
        return results;

    TreeNodeItem* pRootItem = static_cast<TreeNodeItem*>(itemFromIndex(root));
    ZASSERT_EXIT(pRootItem, results);

    for (int row = 0; row < pRootItem->rowCount(); row++)
    {
        QStandardItem* pChildItem = pRootItem->child(row);
        if (!pChildItem)
            continue;
        bool bAppend = false;
        if (SEARCH_SUBNET & searchType)
        {
            if (pChildItem->data(ROLE_NODETYPE).toInt() == SUBGRAPH_NODE)
                bAppend = search_result(root, pChildItem->index(), content, SEARCH_NODECLS, searchOpts, results);
        }
        if (!bAppend && SEARCH_NODEID & searchType)
        {
            bAppend = search_result(root, pChildItem->index(), content, SEARCH_NODEID, searchOpts, results);
        }
        if (!bAppend && (SEARCH_NODECLS & searchType))
        {
            bAppend = search_result(root, pChildItem->index(), content, SEARCH_NODECLS, searchOpts, results);
        }
        if (!bAppend && (searchType & SEARCH_CUSTOM_NAME)) 
        {
            bAppend = search_result(root, pChildItem->index(), content, SEARCH_CUSTOM_NAME, searchOpts, results);
        }
        if (!bAppend && (searchType & SEARCH_ARGS)) 
        {
            bAppend = search_result(root, pChildItem->index(), content, SEARCH_ARGS, searchOpts, results);
        }
        if (bRecursivly && pChildItem->rowCount() > 0) {
             results.append(search_impl(pChildItem->index(), content, searchType, searchOpts, bRecursivly));
        }
    }
    return results;
}

bool GraphsTreeModel_impl::search_result(const QModelIndex& root, const QModelIndex& index, const QString& content, int searchType, int searchOpts, QList<SEARCH_RESULT>& results)
{
    bool ret = false;
    if (SEARCH_ARGS == searchType)
    {
        INPUT_SOCKETS inputs = index.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
        for (const auto& input : inputs)
        {
            QVariant val = input.second.info.defaultValue;
            if (val.type() == QVariant::String)
            {
                QString str = val.toString();
                if ((searchOpts == SEARCH_MATCH_EXACTLY && str == content)
                    || (searchOpts != SEARCH_MATCH_EXACTLY && str.contains(content, Qt::CaseInsensitive))) {
                    ret = true;
                    break;
                }
            }
        }
        if (!ret)
        {
            PARAMS_INFO params = index.data(ROLE_PARAMETERS).value<PARAMS_INFO>();
            for (const auto& param : params)
            {
                QVariant val = param.defaultValue;
                if (val.type() == QVariant::String)
                {
                    QString str = val.toString();
                    if ((searchOpts == SEARCH_MATCH_EXACTLY && str == content)
                        || (searchOpts != SEARCH_MATCH_EXACTLY && str.contains(content, Qt::CaseInsensitive))) {
                        ret = true;
                        break;
                    }
                }
            }
        }
    }
    else
    {
        QString str;
        if (SEARCH_NODEID == searchType)
        {
            str = index.data(ROLE_OBJID).toString();
        }
        else if (SEARCH_NODECLS)
        {
            str = index.data(ROLE_OBJNAME).toString();
        }
        else if (SEARCH_CUSTOM_NAME == searchType)
        {
            str = index.data(ROLE_CUSTOM_OBJNAME).toString();
        }
        if ((searchOpts == SEARCH_MATCH_EXACTLY && str == content)
            || (searchOpts != SEARCH_MATCH_EXACTLY && str.contains(content, Qt::CaseInsensitive))) {
            ret = true;
        }
    }
    if (ret)
    {
        SEARCH_RESULT result;
        result.targetIdx = index;
        result.subgIdx = root;
        result.type = (SearchType)searchType;
        results.append(result);
    }
    return ret;
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

void GraphsTreeModel_impl::renameSubGraph(const QString &oldName, const QString &newName) {
    if (m_treeNodeItems.find(oldName) != m_treeNodeItems.end()) {
        for (auto item : m_treeNodeItems[oldName]) {
            item->setData(newName, ROLE_OBJNAME);
        }

        m_treeNodeItems[newName] = m_treeNodeItems[oldName];
        m_treeNodeItems.remove(oldName);
    }
}

void GraphsTreeModel_impl::appendSubGraphNode(TreeNodeItem *pSubgraph) 
{
    if (!IsSubGraphNode(pSubgraph->index()))
        return;
    QString nodeCls = pSubgraph->objClass();
    if (m_treeNodeItems.find(nodeCls) == m_treeNodeItems.end())
        m_treeNodeItems[nodeCls] = QList<TreeNodeItem *>();
    m_treeNodeItems[nodeCls] << pSubgraph;
    for (int row = 0; row < pSubgraph->rowCount(); row++) {
        TreeNodeItem *pChildItem = static_cast<TreeNodeItem *>(pSubgraph->child(row));
        ZASSERT_EXIT(pChildItem);
        appendSubGraphNode(pChildItem);
    }
}

void GraphsTreeModel_impl::removeSubGraphNode(TreeNodeItem *pSubgraph) 
{
    QString nodeCls = pSubgraph->objClass();
    if (m_treeNodeItems.contains(nodeCls)) {
        m_treeNodeItems[nodeCls].removeOne(pSubgraph);
    }
    for (int row = 0; row < pSubgraph->rowCount(); row++) {
        TreeNodeItem *pChildItem = static_cast<TreeNodeItem *>(pSubgraph->child(row));
        ZASSERT_EXIT(pChildItem);
        removeSubGraphNode(pChildItem);
    }
}

void GraphsTreeModel_impl::onSubgrahSync(const QModelIndex& subgIdx) 
{
    QString nodeCls = subgIdx.data(ROLE_OBJNAME).toString();
    if (m_treeNodeItems.find(nodeCls) != m_treeNodeItems.end()) {
        for (auto pItem : m_treeNodeItems[nodeCls])
        {
            if (subgIdx == pItem->index())
            {
                continue;
            }
            //delete old child items
            while (pItem->rowCount() > 0)
            {
                TreeNodeItem *pChildItem = static_cast<TreeNodeItem*>(pItem->child(0));
                ZASSERT_EXIT(pChildItem);
                QString ident = pChildItem->data(ROLE_OBJID).toString();
                removeNode(ident, pItem->index(), true);
            }
            LINKS_DATA links;
            NODES_DATA childrens = NodesMgr::getChildItems(pItem->parent()->index(), nodeCls, pItem->objName(), links);
            importNodes(childrens, links, QPointF(), pItem->index(), true);

        }
    }
}
QModelIndex GraphsTreeModel_impl::extractSubGraph(const QModelIndexList& nodesIndice, const QModelIndexList& links, const QModelIndex& fromSubgIdx, const QString& toSubg, bool enableTrans)
{
    GraphsModel* pModel = qobject_cast<GraphsModel*>(zenoApp->graphsManagment()->sharedSubgraphs());
    ZASSERT_EXIT(pModel, QModelIndex());

    if (nodesIndice.isEmpty() || !fromSubgIdx.isValid() || toSubg.isEmpty() || pModel->subGraph(toSubg))
    {
        return QModelIndex();
    }

    enableTrans = true;
    if (enableTrans)
        pModel->beginTransaction("extract a new graph");

    //first, new the target subgraph
    pModel->newSubgraph(toSubg);
    QModelIndex toSubgIdx = pModel->index(toSubg);

    //copy nodes to new subg.
    QPair<NODES_DATA, LINKS_DATA> datas = UiHelper::dumpNodes(nodesIndice, links);
    QMap<QString, NODE_DATA> newNodes;
    QList<EdgeInfo> newLinks;
    UiHelper::reAllocIdents(toSubg, datas.first, datas.second, newNodes, newLinks);

    //paste nodes on new subgraph.
    pModel->importNodes(newNodes, newLinks, QPointF(0, 0), toSubgIdx, true);

    //remove nodes from old subg.
    QStringList ids;
    for (QModelIndex idx : nodesIndice)
        ids.push_back(idx.data(ROLE_OBJID).toString());
    for (QString id : ids)
        removeNode(id, fromSubgIdx, enableTrans);

    if (enableTrans)
        pModel->endTransaction();

    return toSubgIdx;
}