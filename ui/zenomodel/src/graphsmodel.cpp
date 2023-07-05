#include "subgraphmodel.h"
#include "graphsmodel.h"
#include "modelrole.h"
#include "uihelper.h"
#include "nodesmgr.h"
#include "zassert.h"
#include <zeno/zeno.h>
#include "common_def.h"
#include <zeno/utils/scope_exit.h>
#include "variantptr.h"
#include "apilevelscope.h"
#include "globalcontrolmgr.h"
#include "dictkeymodel.h"
#include "graphsmanagment.h"
#include <zenoedit/zenoapplication.h>


GraphsModel::GraphsModel(QObject *parent)
    : IGraphsModel(parent)
    , m_selection(nullptr)
    , m_dirty(false)
    , m_stack(new QUndoStack(this))
    , m_apiLevel(0)
    , m_bIOProcessing(false)
    , m_version(zenoio::VER_2_5)
    , m_bApiEnableRun(true)
{
    m_selection = new QItemSelectionModel(this);
}

GraphsModel::~GraphsModel()
{
    clear();
}

QItemSelectionModel* GraphsModel::selectionModel() const
{
    return m_selection;
}

SubGraphModel* GraphsModel::subGraph(const QString& name) const
{
    auto iter = m_subGraphs.find(name);
    if (iter == m_subGraphs.end())
        return nullptr;

    return iter.value();
}

SubGraphModel* GraphsModel::subGraph(int idx) const
{
    if (idx >= 0 && idx < m_subGraphs.count())
    {
        auto itRow = m_row2Key.find(idx);
        ZASSERT_EXIT(itRow != m_row2Key.end(), nullptr);
        QString subgName = itRow.value();
        return subGraph(subgName);
    }
    return nullptr;
}

SubGraphModel* GraphsModel::currentGraph()
{
    return subGraph(m_selection->currentIndex().row());
}

void GraphsModel::switchSubGraph(const QString& graphName)
{
    QModelIndex startIndex = createIndex(0, 0, nullptr);
    const QModelIndexList &lst = this->match(startIndex, ROLE_OBJNAME, graphName, 1, Qt::MatchExactly);
    if (lst.size() == 1)
    {
        m_selection->setCurrentIndex(lst[0], QItemSelectionModel::Current);
    }
}

void GraphsModel::initMainGraph()
{
    SubGraphModel* subGraphModel = new SubGraphModel(this);
    subGraphModel->setName("main");
    appendSubGraph(subGraphModel);
}

void GraphsModel::newSubgraph(const QString &graphName)
{
    //todo: ret value
    if (graphName.compare("main", Qt::CaseInsensitive) == 0)
    {
        zeno::log_error("main graph is not allowed to be created or removed");
        return;
    }

    if (m_subGraphs.contains(graphName))
    {
        zeno::log_error("Already has a graph or node called \"{}\"", graphName.toStdString());
        return;
    }

    QModelIndex startIndex = createIndex(0, 0, nullptr);
    const QModelIndexList &lst = this->match(startIndex, ROLE_OBJNAME, graphName, 1, Qt::MatchExactly);
    if (lst.size() == 1)
    {
        m_selection->setCurrentIndex(lst[0], QItemSelectionModel::Current);
    }
    else
    {
        SubGraphModel *subGraphModel = new SubGraphModel(this);
        subGraphModel->setName(graphName);
        appendSubGraph(subGraphModel);
        m_selection->setCurrentIndex(index(rowCount() - 1, 0), QItemSelectionModel::Current);
        markDirty();
    }
}

void GraphsModel::renameSubGraph(const QString& oldName, const QString& newName)
{
    if (oldName == newName || oldName.compare("main", Qt::CaseInsensitive) == 0)
        return;

    //todo: transaction.
    SubGraphModel* pSubModel = subGraph(oldName);
    ZASSERT_EXIT(pSubModel);
    pSubModel->setName(newName);

    for (int r = 0; r < this->rowCount(); r++)
    {
        SubGraphModel* pModel = subGraph(r);
        ZASSERT_EXIT(pModel);
        const QString& subgraphName = pModel->name();
        if (subgraphName != oldName)
        {
            pModel->replaceSubGraphNode(oldName, newName);
        }
    }

    //replace:
    ZASSERT_EXIT(m_subGraphs.find(oldName) != m_subGraphs.end());
    m_subGraphs.remove(oldName);
    m_subGraphs.insert(newName, pSubModel);

    ZASSERT_EXIT(m_linksGroup.find(oldName) != m_linksGroup.end());
    LinkModel* pLinkModel = m_linksGroup[oldName];
    m_linksGroup.remove(oldName);
    m_linksGroup.insert(newName, pLinkModel);

    auto& mgr = GraphsManagment::instance();
    mgr.renameSubGraph(oldName, newName);

    int row = m_key2Row[oldName];
    m_key2Row[newName] = row;
    m_key2Row.remove(oldName);
    m_row2Key[row] = newName;

    uint32_t ident = m_name2id[oldName];
    m_id2name[ident] = newName;
    ZASSERT_EXIT(m_name2id.find(oldName) != m_name2id.end());
    m_name2id.remove(oldName);
    m_name2id[newName] = ident;

    emit graphRenamed(oldName, newName);
}

QModelIndex GraphsModel::nodeIndex(uint32_t sid, uint32_t nodeid)
{
    auto iter = m_id2name.find(sid);
    if (iter == m_id2name.end())
        return QModelIndex();

    const QString& subgName = iter.value();
    auto iter2 = m_subGraphs.find(subgName);
    if (iter2 == m_subGraphs.end())
    {
        return QModelIndex();
    }

    SubGraphModel* pSubg = iter2.value();
    ZASSERT_EXIT(pSubg, QModelIndex());
    return pSubg->index(nodeid);
}

QModelIndex GraphsModel::subgIndex(uint32_t sid)
{
    ZASSERT_EXIT(m_id2name.find(sid) != m_id2name.end(), QModelIndex());
    const QString& subgName = m_id2name[sid];
    return index(subgName);
}

QModelIndex GraphsModel::_createIndex(SubGraphModel* pSubModel) const
{
    if (!pSubModel)
        return QModelIndex();

    const QString& subgName = pSubModel->name();
    ZASSERT_EXIT(m_name2id.find(subgName) != m_name2id.end(), QModelIndex());
    ZASSERT_EXIT(m_key2Row.find(subgName) != m_key2Row.end(), QModelIndex());
    int row = m_key2Row[subgName];
    uint32_t uuid = m_name2id[subgName];
    return createIndex(row, 0, uuid);
}

QModelIndex GraphsModel::index(int row, int column, const QModelIndex& parent) const
{
    if (row < 0 || row >= m_subGraphs.size())
        return QModelIndex();

    ZASSERT_EXIT(m_row2Key.find(row) != m_row2Key.end(), QModelIndex());
    const QString& subgName = m_row2Key[row];
    uint32_t uuid = m_name2id[subgName];
    return createIndex(row, 0, uuid);
}

QModelIndex GraphsModel::index(const QString& subGraphName) const
{
    auto itRow = m_key2Row.find(subGraphName);
    if (itRow == m_key2Row.end())
        return QModelIndex();
    int row = itRow.value();
    uint32_t uuid = m_name2id[subGraphName];
    return createIndex(row, 0, uuid);
}

QModelIndex GraphsModel::indexBySubModel(SubGraphModel* pSubModel) const
{
    return _createIndex(pSubModel);
}

QModelIndex GraphsModel::linkIndex(const QModelIndex& subgIdx, int r)
{
    LinkModel *pLinkModel = linkModel(subgIdx);
    ZASSERT_EXIT(pLinkModel, QModelIndex());
    return pLinkModel->index(r, 0);
}

QModelIndex GraphsModel::linkIndex(const QModelIndex &subgIdx, 
                                   const QString& outNode,
                                   const QString& outSock,
                                   const QString& inNode,
                                   const QString& inSock)
{
    LinkModel* pLinkModel = linkModel(subgIdx);
    ZASSERT_EXIT(pLinkModel, QModelIndex());
    if (pLinkModel == nullptr)
        return QModelIndex();
    for (int r = 0; r < pLinkModel->rowCount(); r++)
    {
        QModelIndex idx = pLinkModel->index(r, 0);
        if (outNode == idx.data(ROLE_OUTNODE).toString() &&
            outSock == idx.data(ROLE_OUTSOCK).toString() &&
            inNode == idx.data(ROLE_INNODE).toString() &&
            inSock == idx.data(ROLE_INSOCK).toString())
        {
            return idx;
        }
    }
    return QModelIndex();
}

QModelIndex GraphsModel::parent(const QModelIndex& child) const
{
    return QModelIndex();
}

QModelIndex GraphsModel::indexFromPath(const QString& path)
{
    QStringList lst = path.split(cPathSeperator, QtSkipEmptyParts);
    if (lst.size() == 1)
    {
        const QString& nodePath = lst[0];
        lst = nodePath.split('/', QtSkipEmptyParts);
        if (lst[0] == "main")
            return zenoApp->graphsManagment()->currentModel()->indexFromPath(path);
        return nodeIndex(lst.last());
    }
    else if (lst.size() == 2)
    {
        const QString& nodePath = lst[0];
        QModelIndex nodeIdx = indexFromPath(nodePath);

        const QString& paramPath = lst[1];
        if (paramPath.startsWith("[node]"))
        {
            const QString& paramObj = paramPath.mid(QString("[node]").length());
            ViewParamModel* viewParams = QVariantPtr<ViewParamModel>::asPtr(nodeIdx.data(ROLE_NODE_PARAMS));
            QModelIndex paramIdx = viewParams->indexFromPath(paramObj);
            return paramIdx;
        }
        else if (paramPath.startsWith("[panel]"))
        {
            const QString& paramObj = paramPath.mid(QString("[panel]").length());
            ViewParamModel* viewParams = QVariantPtr<ViewParamModel>::asPtr(nodeIdx.data(ROLE_PANEL_PARAMS));
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

QVariant GraphsModel::data(const QModelIndex& index, int role) const
{
    if (!index.isValid())
        return QVariant();

    switch (role)
    {
        case Qt::DisplayRole:
        case Qt::EditRole:
        case ROLE_OBJNAME: return m_row2Key[index.row()];
            //return m_subGraphs[index.row()]->name();
        case ROLE_OBJPATH:
        {
            const QString& subgName = m_row2Key[index.row()];
            return subgName;
        }
    }
    return QVariant();
}

int GraphsModel::rowCount(const QModelIndex& parent) const
{
    return m_subGraphs.size();
}

int GraphsModel::columnCount(const QModelIndex& parent) const
{
    return 1;
}

Qt::ItemFlags GraphsModel::flags(const QModelIndex& index) const
{
    return IGraphsModel::flags(index) | Qt::ItemIsEditable;
}

bool GraphsModel::setData(const QModelIndex& index, const QVariant& value, int role)
{
	if (role == Qt::EditRole)
	{
		const QString& newName = value.toString();
		const QString& oldName = data(index, Qt::DisplayRole).toString();
		if (newName != oldName)
		{
			SubGraphModel* pModel = subGraph(oldName);
			if (!oldName.isEmpty())
			{
				renameSubGraph(oldName, newName);
			}
		}
	} 
	return false;
}

void GraphsModel::revert(const QModelIndex& idx)
{
	const QString& subgName = idx.data().toString();
	if (subgName.isEmpty())
	{
		//exitting new item
        removeGraph(idx.row());
	}
}

void GraphsModel::parseDescStr(const QString& descStr, QString& name, QString& type, QVariant& defl)
{
    auto _arr = descStr.split('@', QtSkipEmptyParts);
    ZASSERT_EXIT(!_arr.isEmpty());

    if (_arr.size() == 1)
    {
        name = _arr[0];
    }
    else if (_arr.size() == 2)
    {
        type = _arr[0];
        name = _arr[1];
        if (type == "string")
            defl = UiHelper::parseStringByType("", type);
    }
    else if (_arr.size() == 3)
    {
        type = _arr[0];
        name = _arr[1];
        QString strDefl = _arr[2];
        defl = UiHelper::parseStringByType(strDefl, type);
    }
}

NODE_DESCS GraphsModel::getCoreDescs()
{
    NODE_DESCS descs;
    QString strDescs = QString::fromStdString(zeno::getSession().dumpDescriptors());
    //zeno::log_critical("EEEE {}", strDescs.toStdString());
    //ZENO_P(strDescs.toStdString());
	QStringList L = strDescs.split("\n");
	for (int i = 0; i < L.size(); i++)
	{
		QString line = L[i];
		if (line.startsWith("DESC@"))
		{
			line = line.trimmed();
			int idx1 = line.indexOf("@");
			int idx2 = line.indexOf("@", idx1 + 1);
			ZASSERT_EXIT(idx1 != -1 && idx2 != -1, descs);
			QString wtf = line.mid(0, idx1);
			QString z_name = line.mid(idx1 + 1, idx2 - idx1 - 1);
			QString rest = line.mid(idx2 + 1);
			ZASSERT_EXIT(rest.startsWith("{") && rest.endsWith("}"), descs);
			auto _L = rest.mid(1, rest.length() - 2).split("}{");
			QString inputs = _L[0], outputs = _L[1], params = _L[2], categories = _L[3];
			QStringList z_categories = categories.split('%', QtSkipEmptyParts);

            NODE_DESC desc;
            for (QString input : inputs.split("%", QtSkipEmptyParts))
            {
                QString type, name;
                QVariant defl;

                parseDescStr(input, name, type, defl);

                INPUT_SOCKET socket;
                socket.info.type = type;
                socket.info.name = name;
                CONTROL_INFO ctrlInfo = UiHelper::getControlByType(z_name, PARAM_INPUT, name, type);
                socket.info.control = ctrlInfo.control;
                socket.info.ctrlProps = ctrlInfo.controlProps.toMap();
                socket.info.defaultValue = defl;
                desc.inputs[name] = socket;
            }
            for (QString output : outputs.split("%", QtSkipEmptyParts))
            {
                QString type, name;
                QVariant defl;

                parseDescStr(output, name, type, defl);

                OUTPUT_SOCKET socket;
                socket.info.type = type;
                socket.info.name = name;
                CONTROL_INFO ctrlInfo = UiHelper::getControlByType(z_name, PARAM_OUTPUT, name, type);
                socket.info.control = ctrlInfo.control;
                socket.info.ctrlProps = ctrlInfo.controlProps.toMap();
                socket.info.defaultValue = defl;
                desc.outputs[name] = socket;
            }
            for (QString param : params.split("%", QtSkipEmptyParts))
            {
                QString type, name;
                QVariant defl;

                parseDescStr(param, name, type, defl);

                PARAM_INFO paramInfo;
                paramInfo.bEnableConnect = false;
                paramInfo.name = name;
                paramInfo.typeDesc = type;
                CONTROL_INFO ctrlInfo = UiHelper::getControlByType(z_name, PARAM_PARAM, name, type);
                paramInfo.control = ctrlInfo.control;
                paramInfo.controlProps = ctrlInfo.controlProps;
                paramInfo.defaultValue = defl;
                //thers is no "value" in descriptor, but it's convient to initialize param value. 
                paramInfo.value = paramInfo.defaultValue;
                desc.params[name] = paramInfo;
            }
            desc.categories = z_categories;
            desc.name = z_name;

            descs.insert(z_name, desc);
        }
    }
    return descs;
}

NODE_DESC GraphsModel::getSubgraphDesc(SubGraphModel* pModel)
{
    const QString& graphName = pModel->name();
    if (graphName == "main" || graphName.isEmpty())
        return NODE_DESC();

    QString subcategory = "subgraph";
    INPUT_SOCKETS subInputs;
    OUTPUT_SOCKETS subOutputs;
    for (int i = 0; i < pModel->rowCount(); i++)
    {
        QModelIndex idx = pModel->index(i, 0);
        const QString& nodeName = idx.data(ROLE_OBJNAME).toString();
        PARAMS_INFO params = idx.data(ROLE_PARAMETERS).value<PARAMS_INFO>();
        if (nodeName == "SubInput")
        {
            QString n_type = params["type"].value.toString();
            QString n_name = params["name"].value.toString();
            auto n_defl = params["defl"].value;

            SOCKET_INFO info;
            info.name = n_name;
            info.type = n_type;
            info.defaultValue = n_defl; 

            INPUT_SOCKET inputSock;
            inputSock.info = info;
            subInputs.insert(n_name, inputSock);
        }
        else if (nodeName == "SubOutput")
        {
            auto n_type = params["type"].value.toString();
            auto n_name = params["name"].value.toString();
            auto n_defl = params["defl"].value;

            SOCKET_INFO info;
            info.name = n_name;
            info.type = n_type;
            info.defaultValue = n_defl; 

            OUTPUT_SOCKET outputSock;
            outputSock.info = info;
            subOutputs.insert(n_name, outputSock);
        }
        else if (nodeName == "SubCategory")
        {
            subcategory = params["name"].value.toString();
        }
    }

    INPUT_SOCKET srcSock;
    srcSock.info.name = "SRC";
    OUTPUT_SOCKET dstSock;
    dstSock.info.name = "DST";

    subInputs.insert("SRC", srcSock);
    subOutputs.insert("DST", dstSock);

    NODE_DESC desc;
    desc.inputs = subInputs;
    desc.outputs = subOutputs;
    desc.categories.push_back(subcategory);
    desc.is_subgraph = true;
    desc.name = graphName;

    return desc;
}

void GraphsModel::appendSubGraph(SubGraphModel* pGraph)
{
    int row = m_subGraphs.size();
    beginInsertRows(QModelIndex(), row, row);

    const QString& name = pGraph->name();

    int nRows = m_subGraphs.size();
    m_row2Key[nRows] = name;
    m_key2Row[name] = nRows;

    QUuid uuid = QUuid::createUuid();
    uint32_t ident = uuid.data1;
    m_id2name[ident] = name;
    m_name2id[name] = ident;

    //insert
    m_subGraphs.insert(name, pGraph);

    auto iterGroup = m_linksGroup.find(name);
    if (iterGroup == m_linksGroup.end())
    {
        LinkModel* pLinkModel = new LinkModel(this);

        //connect(pLinkModel, &QAbstractItemModel::dataChanged, this, &GraphsModel::on_linkDataChanged);
        //connect(pLinkModel, &QAbstractItemModel::rowsAboutToBeInserted, this, &GraphsModel::on_linkAboutToBeInserted);
        //connect(pLinkModel, &QAbstractItemModel::rowsInserted, this, &GraphsModel::on_linkInserted);
        //connect(pLinkModel, &QAbstractItemModel::rowsAboutToBeRemoved, this, &GraphsModel::on_linkAboutToBeRemoved);
        //connect(pLinkModel, &QAbstractItemModel::rowsRemoved, this, &GraphsModel::on_linkRemoved);

        m_linksGroup.insert(name, pLinkModel);
    }

    endInsertRows();
    //the subgraph desc has been inited when processing io.
    if (!IsIOProcessing())
    {
        NODE_DESC desc = getSubgraphDesc(pGraph);
        auto &mgr = GraphsManagment::instance();
        mgr.appendSubGraph(desc);
    }
}

void GraphsModel::removeGraph(int idx)
{
    beginRemoveRows(QModelIndex(), idx, idx);

    auto itRow = m_row2Key.find(idx);
    ZASSERT_EXIT(itRow != m_row2Key.end());
    QString descName = itRow.value();

    for (int r = idx + 1; r < rowCount(); r++)
    {
        const QString &key = m_row2Key[r];
        m_row2Key[r - 1] = key;
        m_key2Row[key] = r - 1;
    }

    //const QString& descName = m_subGraphs[idx]->name();
    m_row2Key.remove(rowCount() - 1);
    m_key2Row.remove(descName);
    m_subGraphs.remove(descName);

    auto iterGroup = m_linksGroup.find(descName);
    if (iterGroup != m_linksGroup.end())
    {
        LinkModel *pLinkModel = iterGroup.value();
        m_linksGroup.remove(descName);
        delete pLinkModel;
    }

    ZASSERT_EXIT(m_name2id.find(descName) != m_name2id.end());
    uint32_t ident = m_name2id[descName];
    m_name2id.remove(descName);
    ZASSERT_EXIT(m_id2name.find(ident) != m_id2name.end());
    m_id2name.remove(ident);

    endRemoveRows();

    //if there is a core node shared the same name with this subgraph,
    // it will not be exported because it was omitted at begin.
    auto &mgr = GraphsManagment::instance();
    mgr.removeGraph(descName);

    markDirty();
}

QModelIndex GraphsModel::fork(const QModelIndex& subgIdx, const QModelIndex &subnetNodeIdx)
{
    const QString& subnetName = subnetNodeIdx.data(ROLE_OBJNAME).toString();
    SubGraphModel* pModel = subGraph(subnetName);
    ZASSERT_EXIT(pModel, QModelIndex());

    NODE_DATA subnetData = _fork(subnetName);
    SubGraphModel *pCurrentModel = subGraph(subgIdx.row());
    pCurrentModel->appendItem(subnetData, false);

    QModelIndex newForkNodeIdx = pCurrentModel->index(subnetData.ident);
    return newForkNodeIdx;
}

NODE_DATA GraphsModel::_fork(const QString& forkSubgName)
{
    SubGraphModel* pModel = subGraph(forkSubgName);
    ZASSERT_EXIT(pModel, NODE_DATA());

    QMap<QString, NODE_DATA> nodes;
    QMap<QString, NODE_DATA> oldGraphsToNew;
    QList<EdgeInfo> links;
    for (int r = 0; r < pModel->rowCount(); r++)
    {
        QModelIndex newIdx;
        QModelIndex idx = pModel->index(r, 0);
        NODE_DATA nodeData;
        if (IsSubGraphNode(idx))
        {
            const QString& snodeId = idx.data(ROLE_OBJID).toString();
            const QString& ssubnetName = idx.data(ROLE_OBJNAME).toString();
            SubGraphModel* psSubModel = subGraph(ssubnetName);
            ZASSERT_EXIT(psSubModel, NODE_DATA());
            nodeData = _fork(ssubnetName);
            const QString& subgNewNodeId = nodeData.ident;

            nodes.insert(snodeId, pModel->nodeData(idx));
            oldGraphsToNew.insert(snodeId, nodeData);
        }
        else
        {
            nodeData = pModel->nodeData(idx);
            const QString &ident = idx.data(ROLE_OBJID).toString();
            nodes.insert(ident, nodeData);
        }
    }

    QModelIndex subgIdx = this->index(forkSubgName);
    LinkModel *pLinkModel = linkModel(subgIdx);
    ZASSERT_EXIT(pLinkModel, NODE_DATA());

    for (int r = 0; r < pLinkModel->rowCount(); r++)
    {
        QModelIndex idx = pLinkModel->index(r, 0);
        const QString& outNode = idx.data(ROLE_OUTNODE).toString();
        const QString& inNode = idx.data(ROLE_INNODE).toString();
        if (nodes.find(inNode) != nodes.end() && nodes.find(outNode) != nodes.end())
        {
            QModelIndex outSockIdx = idx.data(ROLE_OUTSOCK_IDX).toModelIndex();
            QModelIndex inSockIdx = idx.data(ROLE_INSOCK_IDX).toModelIndex();
            QString outSockPath = outSockIdx.data(ROLE_OBJPATH).toString();
            QString inSockPath = inSockIdx.data(ROLE_OBJPATH).toString();
            if (oldGraphsToNew.find(inNode) != oldGraphsToNew.end()) {
                QString newId = oldGraphsToNew[inNode].ident;
                QString oldId = UiHelper::getSockNode(inSockPath);
                inSockPath.replace(oldId, newId);
            }
            if (oldGraphsToNew.find(outNode) != oldGraphsToNew.end()) {
                QString newId = oldGraphsToNew[outNode].ident;
                QString oldId = UiHelper::getSockNode(outSockPath);
                outSockPath.replace(oldId, newId);
            }
            links.append(EdgeInfo(outSockPath, inSockPath));
        }
    }
    for (QMap<QString, NODE_DATA>::const_iterator it = oldGraphsToNew.cbegin(); it != oldGraphsToNew.cend(); it++) {
        const QString &ident = it.key();
        if (nodes.find(ident) != nodes.end()) {
            NODE_DATA newData = it.value();
            NODE_DATA oldData = nodes[ident];
            oldData.ident = newData.ident;
            oldData.nodeCls = newData.nodeCls;

            newData = oldData;

            nodes.remove(ident);
            nodes.insert(newData.ident, newData);
        }
    }

    const QString& forkName = uniqueSubgraph(forkSubgName);
    SubGraphModel* pForkModel = new SubGraphModel(this);
    pForkModel->setName(forkName);
    appendSubGraph(pForkModel);

    NODES_DATA newNodes;
    LINKS_DATA newLinks;

    UiHelper::reAllocIdents(forkName, nodes, links, /*oldGraphsToNew*/ newNodes, newLinks);

    QModelIndex newSubgIdx = indexBySubModel(pForkModel);

    // import new nodes and links into the new created subgraph.
    importNodes(newNodes, newLinks, QPointF(), newSubgIdx, false);

    //create the new fork subnet node at outter layer.
    NODE_DATA subnetData = NodesMgr::newNodeData(this, subgIdx, forkSubgName);
    subnetData.ident = UiHelper::generateUuid(forkName);
    subnetData.nodeCls = forkName;
    //clear the link.
    for (auto it = subnetData.outputs.begin(); it != subnetData.outputs.end(); it++) {
        it->second.info.links.clear();
    }
    for (auto it = subnetData.inputs.begin(); it != subnetData.inputs.end(); it++) {
        it->second.info.links.clear();
    }

    //temp code: node pos.
    QPointF pos = subnetData.pos;
    pos.setY(pos.y() + 100);
    subnetData.pos = pos;
    return subnetData;
}

QString GraphsModel::uniqueSubgraph(QString orginName)
{
    QString newSubName = orginName;
    while (subGraph(newSubName)) {
        newSubName = UiHelper::nthSerialNumName(newSubName);
    }
    return newSubName;
}

void GraphsModel::onCurrentIndexChanged(int row)
{
    const QString& graphName = data(index(row, 0), ROLE_OBJNAME).toString();
    switchSubGraph(graphName);
}

bool GraphsModel::isDirty() const
{
    return m_dirty;
}

void GraphsModel::markDirty()
{
    m_dirty = true;
    emit dirtyChanged();
}

void GraphsModel::clearDirty()
{
    m_dirty = false;
    emit dirtyChanged();
}

void GraphsModel::onRemoveCurrentItem()
{
    removeGraph(m_selection->currentIndex().row());
    //switch to main.
    QModelIndex startIndex = createIndex(0, 0, nullptr);
    //to ask: if not main scene?
    const QModelIndexList &lst = this->match(startIndex, ROLE_OBJNAME, "main", 1, Qt::MatchExactly);
    if (lst.size() == 1)
        m_selection->setCurrentIndex(index(lst[0].row(), 0), QItemSelectionModel::Current);
}

void GraphsModel::beginTransaction(const QString& name)
{
    m_stack->beginMacro(name);
    beginApiLevel();
}

void GraphsModel::endTransaction()
{
    m_stack->endMacro();
    endApiLevel();
}

void GraphsModel::beginApiLevel()
{
    if (IsIOProcessing() || !isApiRunningEnable())
        return;

    //todo: Thread safety
    m_apiLevel++;
}

void GraphsModel::endApiLevel()
{
    if (IsIOProcessing() || !isApiRunningEnable())
        return;

    m_apiLevel--;
    if (m_apiLevel == 0)
    {
        onApiBatchFinished();
    }
}

void GraphsModel::undo()
{
    ApiLevelScope batch(this);
    m_stack->undo();
}

void GraphsModel::redo()
{
    ApiLevelScope batch(this);
    m_stack->redo();
}

void GraphsModel::onApiBatchFinished()
{
    emit apiBatchFinished();
}

QModelIndex GraphsModel::nodeIndex(const QString& ident)
{
    for (auto subg : m_subGraphs)
    {
        QModelIndex idx = subg->index(ident, QModelIndex());
        if (idx.isValid())
            return idx;
    }
    return QModelIndex();
}

QModelIndex GraphsModel::index(const QString& id, const QModelIndex& subGpIdx)
{
    SubGraphModel* pGraph = subGraph(subGpIdx.row());
    ZASSERT_EXIT(pGraph, QModelIndex());
    // index of SubGraph rather than Graphs.
    return pGraph->index(id);
}

QModelIndex GraphsModel::index(int r, const QModelIndex& subGpIdx)
{
	SubGraphModel* pGraph = subGraph(subGpIdx.row());
    ZASSERT_EXIT(pGraph, QModelIndex());
	// index of SubGraph rather than Graphs.
	return pGraph->index(r, 0);
}

int GraphsModel::itemCount(const QModelIndex& subGpIdx) const
{
	SubGraphModel* pGraph = subGraph(subGpIdx.row());
    ZASSERT_EXIT(pGraph, 0);
    return pGraph->rowCount();
}

void GraphsModel::addNode(const NODE_DATA& nodeData, const QModelIndex& subGpIdx, bool enableTransaction)
{
    //TODO: add this at the beginning of all apis?
    bool bEnableIOProc = IsIOProcessing();
    if (bEnableIOProc)
        enableTransaction = false;

    if (enableTransaction)
    {
        QString id = nodeData.ident;
        AddNodeCommand* pCmd = new AddNodeCommand(id, nodeData, this, subGpIdx);
        m_stack->push(pCmd);
    }
    else
    {
        ApiLevelScope batch(this);

        SubGraphModel* pGraph = subGraph(subGpIdx.row());
        ZASSERT_EXIT(pGraph);
        if (onSubIOAdd(pGraph, nodeData))
            return;
        if (onListDictAdd(pGraph, nodeData))
            return;
        pGraph->appendItem(nodeData);
    }
}

void GraphsModel::removeNode(const QString& nodeid, const QModelIndex& subGpIdx, bool enableTransaction)
{
	SubGraphModel* pGraph = subGraph(subGpIdx.row());
    ZASSERT_EXIT(pGraph);

    bool bEnableIOProc = IsIOProcessing();
    if (bEnableIOProc)
        enableTransaction = false;

    if (enableTransaction)
    {
        QModelIndex idx = pGraph->index(nodeid);
        int row = idx.row();
        const NODE_DATA& data = pGraph->nodeData(idx);

        RemoveNodeCommand* pCmd = new RemoveNodeCommand(row, data, this, subGpIdx);
        m_stack->push(pCmd);
    }
    else
    {
        ApiLevelScope batch(this);

        const QModelIndex &idx = pGraph->index(nodeid);
        const QString &objName = idx.data(ROLE_OBJNAME).toString();
        if (!bEnableIOProc)
        {
            //if subnode removed, the whole graphs referred to it should be update.
            bool bInserted = false;
            if (objName == "SubInput")
            {
                onSubIOAddRemove(pGraph, idx, true, bInserted);
            }
            else if (objName == "SubOutput")
            {
                onSubIOAddRemove(pGraph, idx, false, bInserted);
            }
        }
        pGraph->removeNode(nodeid, false);
    }
}

void GraphsModel::appendNodes(const QList<NODE_DATA>& nodes, const QModelIndex& subGpIdx, bool enableTransaction)
{
	SubGraphModel* pGraph = subGraph(subGpIdx.row());
    ZASSERT_EXIT(pGraph);
    for (const NODE_DATA& nodeData : nodes)
    {
        addNode(nodeData, subGpIdx, enableTransaction);
    }
}

void GraphsModel::importNodes(
                const QMap<QString, NODE_DATA>& nodes,
                const QList<EdgeInfo>& links,
                const QPointF& pos,
                const QModelIndex& subGpIdx,
                bool enableTransaction)
{
    if (nodes.isEmpty()) return;
    //ZASSERT_EXIT(!nodes.isEmpty());
    if (enableTransaction)
    {
        ImportNodesCommand *pCmd = new ImportNodesCommand(nodes, links, pos, this, subGpIdx);
        m_stack->push(pCmd);
    }
    else
    {
        ApiLevelScope batch(this);

        SubGraphModel* pGraph = subGraph(subGpIdx.row());
        ZASSERT_EXIT(pGraph);
        for (const NODE_DATA& data : nodes)
        {
            addNode(data, subGpIdx, false);
        }

        //resolve pos and links.
        QStringList ids = nodes.keys();
        QModelIndex nodeIdx = index(ids[0], subGpIdx);
        QPointF _pos = nodeIdx.data(ROLE_OBJPOS).toPointF();
        const QPointF offset = pos - _pos;

        for (const QString& ident : ids)
	    {
		    const QModelIndex& idx = pGraph->index(ident);
		    _pos = idx.data(ROLE_OBJPOS).toPointF();
		    _pos += offset;
		    pGraph->setData(idx, _pos, ROLE_OBJPOS);
	    }
        for (EdgeInfo link : links)
        {
            addLink(subGpIdx, link, false);
        }
    }
}

QModelIndex GraphsModel::extractSubGraph(
                    const QModelIndexList& nodesIndice,
                    const QModelIndexList& links,
                    const QModelIndex& fromSubgIdx,
                    const QString& toSubg,
                    bool enableTrans)
{
    if (nodesIndice.isEmpty() || !fromSubgIdx.isValid() || toSubg.isEmpty() || subGraph(toSubg))
    {
        return QModelIndex();
    }

    enableTrans = true;
    if (enableTrans)
        beginTransaction("extract a new graph");

    //first, new the target subgraph
    newSubgraph(toSubg);
    QModelIndex toSubgIdx = index(toSubg);

    //copy nodes to new subg.
    QPair<NODES_DATA, LINKS_DATA> datas = UiHelper::dumpNodes(nodesIndice, links);
    QMap<QString, NODE_DATA> newNodes;
    QList<EdgeInfo> newLinks;
    UiHelper::reAllocIdents(toSubg, datas.first, datas.second, newNodes, newLinks);

    //paste nodes on new subgraph.
    importNodes(newNodes, newLinks, QPointF(0, 0), toSubgIdx, true);

    //remove nodes from old subg.
    QStringList ids;
    for (QModelIndex idx : nodesIndice)
        ids.push_back(idx.data(ROLE_OBJID).toString());
    for (QString id : ids)
        removeNode(id, fromSubgIdx, enableTrans);

    if (enableTrans)
        endTransaction();

    return toSubgIdx;
}

bool GraphsModel::IsSubGraphNode(const QModelIndex& nodeIdx) const
{
    if (!nodeIdx.isValid())
        return false;

    QString nodeName = nodeIdx.data(ROLE_OBJNAME).toString();
    auto &mgr = GraphsManagment::instance();
    NODE_DESC desc;
    if (IsIOProcessing() && mgr.getSubgDesc(nodeName, desc))
    {
        return true;
    }
    return subGraph(nodeName) != nullptr;
}

void GraphsModel::removeNode(int row, const QModelIndex& subGpIdx)
{
	SubGraphModel* pGraph = subGraph(subGpIdx.row());
    ZASSERT_EXIT(pGraph);
	if (pGraph)
	{
        QModelIndex idx = pGraph->index(row, 0);
        pGraph->removeNode(row);
	}
}

void GraphsModel::removeLink(const QModelIndex& linkIdx, bool enableTransaction)
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

void GraphsModel::removeLink(const QModelIndex& subgIdx, const EdgeInfo& link, bool enableTransaction)
{
    if (enableTransaction)
    {
        LinkCommand *pCmd = new LinkCommand(subgIdx, false, link, this);
        m_stack->push(pCmd);
    }
    else
    {
        ApiLevelScope batch(this);

        //sometimes when removing socket, the link attached on it will also be removed,
        //but if the socket index is invalid, then cause the associated link cannot be restored by these sockets.
        //so, we must ensure the removal of link, is ahead of the removal of sockets.

        //find the socket idx
        const QModelIndex& outSockIdx = indexFromPath(link.outSockPath);
        const QModelIndex& inSockIdx = indexFromPath(link.inSockPath);
        ZASSERT_EXIT(outSockIdx.isValid() && inSockIdx.isValid());

        LinkModel *pLinkModel = linkModel(subgIdx);
        ZASSERT_EXIT(pLinkModel);

        //restore the link
        QModelIndex linkIdx = pLinkModel->index(outSockIdx, inSockIdx);

        QAbstractItemModel* pOutputs = const_cast<QAbstractItemModel*>(outSockIdx.model());
        ZASSERT_EXIT(pOutputs);
        pOutputs->setData(outSockIdx, linkIdx, ROLE_REMOVELINK);

        QAbstractItemModel* pInputs = const_cast<QAbstractItemModel*>(inSockIdx.model());
        ZASSERT_EXIT(pInputs);
        pInputs->setData(inSockIdx, linkIdx, ROLE_REMOVELINK);

        ZASSERT_EXIT(linkIdx.isValid());
        pLinkModel->removeRow(linkIdx.row());
    }
}

QModelIndex GraphsModel::addLink(const QModelIndex& subgIdx, const QModelIndex& fromSock, const QModelIndex& toSock, bool enableTransaction)
{
    ZASSERT_EXIT(fromSock.isValid() && toSock.isValid(), QModelIndex());
    EdgeInfo link(fromSock.data(ROLE_OBJPATH).toString(), toSock.data(ROLE_OBJPATH).toString());
    return addLink(subgIdx, link, enableTransaction);
}

QModelIndex GraphsModel::addLink(const QModelIndex& subgIdx, const EdgeInfo& info, bool enableTransaction)
{
    if (enableTransaction)
    {
        LinkCommand *pCmd = new LinkCommand(subgIdx, true, info, this);
        m_stack->push(pCmd);
        //todo: return val on this level.
        return QModelIndex();
    }
    else
    {
        ApiLevelScope batch(this);

        QModelIndex inParamIdx = indexFromPath(info.inSockPath);
        QModelIndex outParamIdx = indexFromPath(info.outSockPath);
        if (!inParamIdx.isValid() || !outParamIdx.isValid())
        {
            QString inSock = UiHelper::getSockNode(info.inSockPath) + "/" + UiHelper::getSockName(info.inSockPath);
            QString outSock = UiHelper::getSockNode(info.outSockPath) + "/" + UiHelper::getSockName(info.outSockPath);
            zeno::log_warn("there is not valid input ({}) or output ({}) sockets.", inSock.toStdString(), outSock.toStdString());
            return QModelIndex();
        }
        if (!subgIdx.isValid())
        {
            zeno::log_warn("addlink: the subgraph has not been specified.");
            return QModelIndex();
        }

        LinkModel *pLinkModel = linkModel(subgIdx);
        ZASSERT_EXIT(pLinkModel, QModelIndex());

        int row = pLinkModel->addLink(outParamIdx, inParamIdx);
        const QModelIndex& linkIdx = pLinkModel->index(row, 0);

        QAbstractItemModel* pInputs = const_cast<QAbstractItemModel*>(inParamIdx.model());
        QAbstractItemModel* pOutputs = const_cast<QAbstractItemModel*>(outParamIdx.model());

        ZASSERT_EXIT(pInputs && pOutputs, QModelIndex());
        pInputs->setData(inParamIdx, linkIdx, ROLE_ADDLINK);
        pOutputs->setData(outParamIdx, linkIdx, ROLE_ADDLINK);
        return linkIdx;
    }
}

void GraphsModel::setIOProcessing(bool bIOProcessing)
{
    m_bIOProcessing = bIOProcessing;
}

bool GraphsModel::IsIOProcessing() const
{
    return m_bIOProcessing;
}

void GraphsModel::removeSubGraph(const QString& name)
{
    if (name.compare("main", Qt::CaseInsensitive) == 0)
        return;

    for (QString subgName : m_subGraphs.keys())
    {
        if (subgName == name) {
            int nRow = m_key2Row[name];
            removeGraph(nRow);
        }
        else
        {
            SubGraphModel* subg = m_subGraphs[subgName];
            ZASSERT_EXIT(subg);
            subg->removeNodeByDescName(name);
        }
    }
}

QModelIndexList GraphsModel::findSubgraphNode(const QString& subgName)
{
    QModelIndexList nodes;
    for (SubGraphModel* subg : m_subGraphs)
    {
        if (subg->name() != subgName)
        {
            nodes.append(subg->getNodesByCls(subgName));
        }
    }
    return nodes;
}

void GraphsModel::updateParamInfo(const QString& id, PARAM_UPDATE_INFO info, const QModelIndex& subGpIdx, bool enableTransaction)
{
    const QModelIndex& nodeIdx = index(id, subGpIdx);
    NodeParamModel* nodeParams = QVariantPtr<NodeParamModel>::asPtr(nodeIdx.data(ROLE_NODE_PARAMS));
    const QModelIndex& paramIdx = nodeParams->getParam(PARAM_PARAM, info.name);
    ModelSetData(paramIdx, info.newValue, ROLE_PARAM_VALUE);
}

void GraphsModel::updateSocketDefl(const QString& id, PARAM_UPDATE_INFO info, const QModelIndex& subGpIdx, bool enableTransaction)
{
    const QModelIndex& nodeIdx = index(id, subGpIdx);
    NodeParamModel* nodeParams = QVariantPtr<NodeParamModel>::asPtr(nodeIdx.data(ROLE_NODE_PARAMS));
    const QModelIndex& paramIdx = nodeParams->getParam(PARAM_INPUT, info.name);
    ModelSetData(paramIdx, info.newValue, ROLE_PARAM_VALUE);
}

int GraphsModel::ModelSetData(
    const QPersistentModelIndex& idx,
    const QVariant& value,
    int role,
    const QString& comment/*todo*/)
{
    if (!idx.isValid())
        return -1;

    QAbstractItemModel* pTargetModel = const_cast<QAbstractItemModel*>(idx.model());
    if (!pTargetModel)
        return -1;

    const QVariant& oldValue = pTargetModel->data(idx, role);
    if (oldValue == value)
        return -1;
    ModelDataCommand* pCmd = new ModelDataCommand(this, idx, oldValue, value, role);
    m_stack->push(pCmd);        //will call model->setData method.
    return 0;
}

int GraphsModel::undoRedo_updateSubgDesc(const QString& descName, const NODE_DESC& desc)
{
    UpdateSubgDescCommand *pCmd = new UpdateSubgDescCommand(this, descName, desc);
    m_stack->push(pCmd);
    return 0;
}

bool GraphsModel::addExecuteCommand(QUndoCommand* pCommand)
{
    //toask: need level?
    if (!pCommand)
        return false;
    m_stack->push(pCommand);
    return 1;
}

void GraphsModel::setIOVersion(zenoio::ZSG_VERSION ver)
{
    m_version = ver;
}

zenoio::ZSG_VERSION GraphsModel::ioVersion() const
{
    return m_version;
}

void GraphsModel::setApiRunningEnable(bool bEnable)
{
    m_bApiEnableRun = bEnable;
}

bool GraphsModel::isApiRunningEnable() const
{
    return m_bApiEnableRun;
}

bool GraphsModel::setCustomName(const QModelIndex &subgIdx, const QModelIndex &index, const QString &value)
{
    QString name = data(subgIdx, Qt::DisplayRole).toString();
    SubGraphModel *pModel = subGraph(name);
    return pModel->setData(index, value, ROLE_CUSTOM_OBJNAME);
}

void GraphsModel::updateNodeStatus(const QString& nodeid, STATUS_UPDATE_INFO info, const QModelIndex& subgIdx, bool enableTransaction)
{
    QModelIndex nodeIdx = index(nodeid, subgIdx);
    ModelSetData(nodeIdx, info.newValue, info.role);
}

void GraphsModel::updateBlackboard(const QString &id, const QVariant &newInfo, const QModelIndex &subgIdx, bool enableTransaction) 
{
    SubGraphModel *pSubg = subGraph(subgIdx.row());
    const QModelIndex& idx = pSubg->index(id);
    ZASSERT_EXIT(pSubg);

    if (enableTransaction)
    {
        if (newInfo.canConvert<BLACKBOARD_INFO>()) 
        {
            PARAMS_INFO params = idx.data(ROLE_PARAMS_NO_DESC).value<PARAMS_INFO>();
            BLACKBOARD_INFO oldInfo = params["blackboard"].value.value<BLACKBOARD_INFO>();
            UpdateBlackboardCommand *pCmd = new UpdateBlackboardCommand(id, newInfo.value<BLACKBOARD_INFO>(), oldInfo, this, subgIdx);
            m_stack->push(pCmd);
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
            pSubg->setData(idx, QVariant::fromValue(params), ROLE_PARAMS_NO_DESC);
        } 
        else if (newInfo.canConvert<STATUS_UPDATE_INFO>()) 
        {
            pSubg->setData(idx, newInfo.value<STATUS_UPDATE_INFO>().newValue, ROLE_OBJPOS);
        }
    }
}

NODE_DATA GraphsModel::itemData(const QModelIndex& index, const QModelIndex& subGpIdx) const
{
	SubGraphModel* pGraph = subGraph(subGpIdx.row());
    ZASSERT_EXIT(pGraph, NODE_DATA());
    return pGraph->nodeData(index);
}

void GraphsModel::exportSubgraph(const QModelIndex& subGpIdx, NODES_DATA& nodes, LINKS_DATA& links) const
{
    SubGraphModel* pGraph = subGraph(subGpIdx.row());
    ZASSERT_EXIT(pGraph);
    for (int r = 0; r < pGraph->rowCount(); r++)
    {
        QModelIndex idx = pGraph->index(r, 0);
        const QString& id = idx.data(ROLE_OBJID).toString();
        nodes[id] = pGraph->nodeData(idx);
    }

    auto lnkModel = linkModel(subGpIdx);
    ZASSERT_EXIT(lnkModel);
    for (int r = 0; r < lnkModel->rowCount(); r++)
    {
        QModelIndex idx = lnkModel->index(r, 0);
        links.append(UiHelper::exportLink(idx));
    }
}

void GraphsModel::setName(const QString& name, const QModelIndex& subGpIdx)
{
	SubGraphModel* pGraph = subGraph(subGpIdx.row());
    ZASSERT_EXIT(pGraph);
    pGraph->setName(name);
}

void GraphsModel::clearSubGraph(const QModelIndex& subGpIdx)
{
	SubGraphModel* pGraph = subGraph(subGpIdx.row());
    ZASSERT_EXIT(pGraph);
    pGraph->clear();
}

void GraphsModel::clear()
{
    for (int r = 0; r < this->rowCount(); r++)
    {
        const QModelIndex& subgIdx = this->index(r, 0);
        clearSubGraph(subgIdx);
    }
    m_linksGroup.clear();
    emit modelClear();
}

QModelIndexList GraphsModel::subgraphsIndice() const
{
    //todo: deprecated
    return QModelIndexList();
}

LinkModel* GraphsModel::linkModel(const QModelIndex& subgIdx) const
{
    const QString &subgName = subgIdx.data(ROLE_OBJNAME).toString();
    auto iterGroup = m_linksGroup.find(subgName);
    ZASSERT_EXIT(iterGroup != m_linksGroup.end(), nullptr);
    LinkModel *pLinkModel = iterGroup.value();
    ZASSERT_EXIT(pLinkModel, nullptr);
    return pLinkModel;
}

QModelIndex GraphsModel::getSubgraphIndex(const QModelIndex& linkIdx)
{
	const QString& inNode = linkIdx.data(ROLE_INNODE).toString();
    QModelIndex inNodeIdx = linkIdx.data(ROLE_INNODE_IDX).toModelIndex();
    if (inNodeIdx.isValid())
    {
        return inNodeIdx.data(ROLE_SUBGRAPH_IDX).toModelIndex();
    }
    return QModelIndex();
}

QRectF GraphsModel::viewRect(const QModelIndex& subgIdx)
{
	SubGraphModel* pModel = subGraph(subgIdx.row());
    ZASSERT_EXIT(pModel, QRectF());
    return pModel->viewRect();
}

void GraphsModel::on_linkDataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles)
{
	const QModelIndex& subgIdx = getSubgraphIndex(topLeft);
    if (subgIdx.isValid())
	    emit linkDataChanged(subgIdx, topLeft, roles[0]);
}

void GraphsModel::on_linkAboutToBeInserted(const QModelIndex& parent, int first, int last)
{
    //LinkModel *pLinkModel = qobject_cast<LinkModel *>(sender());
    //QModelIndex linkIdx = pLinkModel->index(first, 0, parent);
    //if (linkIdx.isValid())
    //{
    //    const QModelIndex &subgIdx = getSubgraphIndex(linkIdx);
    //    if (subgIdx.isValid())
    //        emit linkAboutToBeInserted(subgIdx, parent, first, last);
    //}
}

void GraphsModel::on_linkInserted(const QModelIndex& parent, int first, int last)
{
 //   LinkModel *pLinkModel = qobject_cast<LinkModel *>(sender());
 //   QModelIndex linkIdx = pLinkModel->index(first, 0, parent);
	//const QModelIndex& subgIdx = getSubgraphIndex(linkIdx);
 //   if (subgIdx.isValid())
	//    emit linkInserted(subgIdx, parent, first, last);
}

void GraphsModel::on_linkAboutToBeRemoved(const QModelIndex& parent, int first, int last)
{
 //   LinkModel *pLinkModel = qobject_cast<LinkModel *>(sender());
 //   QModelIndex linkIdx = pLinkModel->index(first, 0, parent);
	//const QModelIndex& subgIdx = getSubgraphIndex(linkIdx);
 //   if (subgIdx.isValid())
	//    emit linkAboutToBeRemoved(subgIdx, parent, first, last);
}

void GraphsModel::on_linkRemoved(const QModelIndex& parent, int first, int last)
{
 //   LinkModel *pLinkModel = qobject_cast<LinkModel *>(sender());
 //   QModelIndex linkIdx = pLinkModel->index(first, 0, parent);
	//const QModelIndex& subgIdx = getSubgraphIndex(linkIdx);
 //   if (subgIdx.isValid())
	//    emit linkRemoved(subgIdx, parent, first, last);
}

bool GraphsModel::onSubIOAdd(SubGraphModel* pGraph, NODE_DATA nodeData2)
{
    const QString& descName = nodeData2.nodeCls;
    if (descName != "SubInput" && descName != "SubOutput")
        return false;

    bool bInput = descName == "SubInput";

    ZASSERT_EXIT(nodeData2.params.find("name") != nodeData2.params.end(), false);
    PARAM_INFO& param = nodeData2.params["name"];
    QString newSockName = UiHelper::correctSubIOName(pGraph, param.value.toString(), bInput);
    param.value = newSockName;
    pGraph->appendItem(nodeData2);

    if (!IsIOProcessing())
    {
        const QModelIndex& nodeIdx = pGraph->index(nodeData2.ident);
        onSubIOAddRemove(pGraph, nodeIdx, bInput, true);
    }
    return true;
}

bool GraphsModel::onListDictAdd(SubGraphModel* pGraph, NODE_DATA nodeData2)
{
    const QString& descName = nodeData2.nodeCls;
    if (descName == "MakeList" || descName == "MakeDict")
    {
        INPUT_SOCKET inSocket;
        inSocket.info.nodeid = nodeData2.ident;

        int maxObjId = UiHelper::getMaxObjId(nodeData2.inputs.keys());
        if (maxObjId == -1)
        {
            inSocket.info.name = "obj0";
            if (descName == "MakeDict")
            {
                inSocket.info.control = CONTROL_NONE;
                inSocket.info.sockProp = SOCKPROP_EDITABLE;
            }
            nodeData2.inputs.insert(inSocket.info.name, inSocket);
        }
        pGraph->appendItem(nodeData2);
        return true;
    }
    else if (descName == "ExtractDict")
    {
        OUTPUT_SOCKET outSocket;
        outSocket.info.nodeid = nodeData2.ident;

        int maxObjId = UiHelper::getMaxObjId(nodeData2.outputs.keys());
        if (maxObjId == -1)
        {
            outSocket.info.name = "obj0";
            outSocket.info.control = CONTROL_NONE;
            outSocket.info.sockProp = SOCKPROP_EDITABLE;
            nodeData2.outputs.insert(outSocket.info.name, outSocket);
        }
        pGraph->appendItem(nodeData2);
        return true;
    }
    return false;
}

void GraphsModel::onSubIOAddRemove(SubGraphModel* pSubModel, const QModelIndex& nodeIdx, bool bInput, bool bInsert)
{
    const QString& objId = nodeIdx.data(ROLE_OBJID).toString();
    const QString& objName = nodeIdx.data(ROLE_OBJNAME).toString();

    NodeParamModel* nodeParams = QVariantPtr<NodeParamModel>::asPtr(nodeIdx.data(ROLE_NODE_PARAMS));

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

    const QString& subnetNodeName = pSubModel->name();

    auto &mgr = GraphsManagment::instance();
    NODE_DESC desc;
    bool ret = mgr.getSubgDesc(subnetNodeName, desc);
    ZASSERT_EXIT(ret);

    SOCKET_INFO info;
    info.control = ctrl;
    info.defaultValue = deflVal;
    info.name = nameValue;
    info.type = typeValue;
    info.ctrlProps = ctrlProps.toMap();
    info.toolTip = toolTip;

    if (bInsert)
    {
        if (bInput)
        {
            if (desc.inputs.find(nameValue) == desc.inputs.end())
                desc.inputs[nameValue] = INPUT_SOCKET();
            desc.inputs[nameValue].info = info;
        }
        else
        {
            if (desc.outputs.find(nameValue) == desc.outputs.end())
                desc.outputs[nameValue] = OUTPUT_SOCKET();
            desc.outputs[nameValue].info = info;
        }

        //sync to all subgraph nodes.
        QModelIndexList subgNodes = findSubgraphNode(subnetNodeName);
        for (const QModelIndex& subgNode : subgNodes)
        {
            NodeParamModel* nodeParams = QVariantPtr<NodeParamModel>::asPtr(subgNode.data(ROLE_NODE_PARAMS));
            nodeParams->setAddParam(
                        bInput ? PARAM_INPUT : PARAM_OUTPUT,
                        nameValue,
                        typeValue,
                        deflVal,
                        ctrl,
                        ctrlProps,
                        SOCKPROP_NORMAL,
                        DICTPANEL_INFO(),
                        toolTip
            );
        }
    }
    else
    {
        if (bInput)
        {
            ZASSERT_EXIT(desc.inputs.find(nameValue) != desc.inputs.end());
            desc.inputs.remove(nameValue);
        }
        else
        {
            ZASSERT_EXIT(desc.outputs.find(nameValue) != desc.outputs.end());
            desc.outputs.remove(nameValue);
        }

        QModelIndexList subgNodes = findSubgraphNode(subnetNodeName);
        for (const QModelIndex& subgNode : subgNodes)
        {
            NodeParamModel* nodeParams = QVariantPtr<NodeParamModel>::asPtr(subgNode.data(ROLE_NODE_PARAMS));
            nodeParams->removeParam(bInput ? PARAM_INPUT : PARAM_OUTPUT, nameValue);
        }
    }
    mgr.updateSubgDesc(subnetNodeName, desc);
}

QList<SEARCH_RESULT> GraphsModel::search(const QString& content, int searchType, int searchOpts)
{
    return search_impl(content, searchType, searchOpts);
}

QModelIndexList GraphsModel::searchInSubgraph(const QString& objName, const QModelIndex& subgIdx)
{
    SubGraphModel* pModel = subGraph(subgIdx.row());
    QVector<SubGraphModel *> vec;
    vec << pModel;
    QList<SEARCH_RESULT> results = search_impl(
                    objName,
                    SEARCH_ARGS | SEARCH_NODECLS | SEARCH_NODEID | SEARCH_CUSTOM_NAME, SEARCH_FUZZ,
                    vec);
    QModelIndexList list;
    for (auto res : results) 
    {
        list.append(res.targetIdx);
    }
    return list;
}

QList<SEARCH_RESULT> GraphsModel::search_impl(
                    const QString &content,
                    int searchType,
                    int searchOpts,
                    QVector<SubGraphModel *> vec)
{
    QList<SEARCH_RESULT> results;
    if (content.isEmpty())
        return results;

    QSet<QString> nodes;
    if (searchType & SEARCH_SUBNET)
    {
        QModelIndexList lst = match(index(0, 0), ROLE_OBJNAME, content, -1, Qt::MatchContains);
        for (const QModelIndex &subgIdx : lst)
        {
            SEARCH_RESULT result;
            result.subgIdx = subgIdx;
            result.targetIdx = subgIdx;
            result.type = SEARCH_SUBNET;
            results.append(result);
        }
    }
    /* start match len*/
    static int sStartMatch = 2;
    if (vec.isEmpty())
    {
        for (auto subg : m_subGraphs)
            vec.append(subg);
    }
    for (auto subgInfo : vec) {
        SubGraphModel *pModel = subgInfo;
        QModelIndex subgIdx = indexBySubModel(pModel);
        if ((searchType & SEARCH_ARGS) && content.size() >= sStartMatch) {
            for (int r = 0; r < pModel->rowCount(); r++) {
                QModelIndex nodeIdx = pModel->index(r, 0);
                INPUT_SOCKETS inputs = nodeIdx.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
                PARAMS_INFO params = nodeIdx.data(ROLE_PARAMETERS).value<PARAMS_INFO>();
                for (const INPUT_SOCKET &inputSock : inputs) {
                    if (inputSock.info.type == "string" || inputSock.info.type == "multiline_string" ||
                        inputSock.info.type == "readpath" || inputSock.info.type == "writepath") {
                        const QString &textValue = inputSock.info.defaultValue.toString();
                        if (textValue.contains(content, Qt::CaseInsensitive)) {
                            SEARCH_RESULT result;
                            result.targetIdx = nodeIdx;
                            result.subgIdx = subgIdx;
                            result.type = SEARCH_ARGS;
                            result.socket = inputSock.info.name;
                            results.append(result);
                            nodes.insert(nodeIdx.data(ROLE_OBJID).toString());
                        }
                    }
                }
                for (const PARAM_INFO &param : params) {
                    if (param.typeDesc == "string" || param.typeDesc == "multiline_string" ||
                        param.typeDesc == "readpath" || param.typeDesc == "writepath") {
                        const QString &textValue = param.value.toString();
                        if (textValue.contains(content, Qt::CaseInsensitive)) {
                            SEARCH_RESULT result;
                            result.targetIdx = nodeIdx;
                            result.subgIdx = subgIdx;
                            result.type = SEARCH_ARGS;
                            result.socket = param.name;
                            results.append(result);
                            nodes.insert(nodeIdx.data(ROLE_OBJID).toString());
                        }
                    }
                }
            }
        }
        if (searchType & SEARCH_NODEID) {
            QModelIndexList lst;
            if (searchOpts == SEARCH_MATCH_EXACTLY) {
                QModelIndex idx = pModel->index(content);
                if (idx.isValid())
                    lst.append(idx);
            } else {
                lst = pModel->match(pModel->index(0, 0), ROLE_OBJID, content, -1, Qt::MatchContains);
            }
            if (!lst.isEmpty()) {
                for (const QModelIndex &nodeIdx : lst) {
                    if (nodes.contains(nodeIdx.data(ROLE_OBJID).toString())) {
                        continue;
                    }
                    SEARCH_RESULT result;
                    result.targetIdx = nodeIdx;
                    result.subgIdx = subgIdx;
                    result.type = SEARCH_NODEID;
                    results.append(result);
                    nodes.insert(nodeIdx.data(ROLE_OBJID).toString());
                }
            }
        }
        if (searchType & SEARCH_NODECLS) {
            QModelIndexList lst = pModel->match(pModel->index(0, 0), ROLE_OBJNAME, content, -1, Qt::MatchContains);
            for (const QModelIndex &nodeIdx : lst) {
                if (nodes.contains(nodeIdx.data(ROLE_OBJID).toString())) {
                    continue;
                }
                QString nodeCls = nodeIdx.data(ROLE_OBJNAME).toString();
                if (searchOpts == SEARCH_MATCH_EXACTLY && nodeCls != content) {
                    continue;
                }
                SEARCH_RESULT result;
                result.targetIdx = nodeIdx;
                result.subgIdx = subgIdx;
                result.type = SEARCH_NODECLS;
                results.append(result);
                nodes.insert(nodeIdx.data(ROLE_OBJID).toString());
            }
        }
        if (searchType & SEARCH_CUSTOM_NAME) {
            QModelIndexList lst = pModel->match(pModel->index(0, 0), ROLE_CUSTOM_OBJNAME, content, -1, Qt::MatchContains);
            for (const QModelIndex &nodeIdx : lst) {
                if (nodes.contains(nodeIdx.data(ROLE_OBJID).toString())) {
                    continue;
                }
                QString customName = nodeIdx.data(ROLE_CUSTOM_OBJNAME).toString();
                if (searchOpts == SEARCH_MATCH_EXACTLY && customName != content) {
                    continue;
                }
                SEARCH_RESULT result;
                result.targetIdx = nodeIdx;
                result.subgIdx = subgIdx;
                result.type = SEARCH_CUSTOM_NAME;
                results.append(result);
                nodes.insert(nodeIdx.data(ROLE_OBJID).toString());
            }
        }
    }
    return results;
}

void GraphsModel::collaspe(const QModelIndex& subgIdx)
{
	SubGraphModel* pModel = subGraph(subgIdx.row());
    ZASSERT_EXIT(pModel);
    pModel->collaspe();
}

void GraphsModel::expand(const QModelIndex& subgIdx)
{
    SubGraphModel* pModel = subGraph(subgIdx.row());
    ZASSERT_EXIT(pModel);
    pModel->expand();
}

void GraphsModel::setNodeData(const QModelIndex &nodeIndex, const QModelIndex &subGpIdx, const QVariant &value, int role) {
    SubGraphModel* pModel = this->subGraph(subGpIdx.row());
    ZASSERT_EXIT(pModel);
    pModel->setData(nodeIndex, value, role);
}

void GraphsModel::onSubgrahSync(const QModelIndex& subgIdx)
{
    IGraphsModel* pModel = UiHelper::getGraphsBySubg(subgIdx);
    ZASSERT_EXIT(pModel);
    updateSubgrahs(subgIdx);
    pModel->onSubgrahSync(subgIdx);
}

void GraphsModel::updateSubgrahs(const QModelIndex& subgIdx)
{
    QString nodeCls = subgIdx.data(ROLE_OBJNAME).toString();
    SubGraphModel* pSubgModel = subGraph(nodeCls);
    if (pSubgModel) {
        //delete old child items
        while (pSubgModel->rowCount() > 0)
        {
            QModelIndex index = pSubgModel->index(0, 0);
            QString ident = index.data(ROLE_OBJID).toString();
            removeNode(ident, this->index(nodeCls), true);
        }
        //import nodes
        NODE_DATA node = subgIdx.data(ROLE_OBJDATA).value<NODE_DATA>();
        LINKS_DATA oldLinks;
        if (IGraphsModel* pModel = UiHelper::getGraphsBySubg(subgIdx))
        {
            for (int i = 0; i < pModel->itemCount(subgIdx); i++)
            {
                const QModelIndex& childIdx = pModel->index(i, subgIdx);
                if (childIdx.isValid())
                {
                    if (pModel->IsSubGraphNode(childIdx))
                    {
                        onSubgrahSync(childIdx);
                    }
                    NodeParamModel* viewParams = QVariantPtr<NodeParamModel>::asPtr(childIdx.data(ROLE_NODE_PARAMS));
                    const QModelIndexList& lst = viewParams->getInputIndice();
                    for (int r = 0; r < lst.size(); r++)
                    {
                        const QModelIndex& paramIdx = lst[r];
                        const QString& inSock = paramIdx.data(ROLE_PARAM_NAME).toString();
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
                            QAbstractItemModel* pKeyObjModel =
                                QVariantPtr<QAbstractItemModel>::asPtr(paramIdx.data(ROLE_VPARAM_LINK_MODEL));
                            for (int _r = 0; _r < pKeyObjModel->rowCount(); _r++)
                            {
                                const QModelIndex& keyIdx = pKeyObjModel->index(_r, 0);
                                ZASSERT_EXIT(keyIdx.isValid());
                                PARAM_LINKS links = keyIdx.data(ROLE_PARAM_LINKS).value<PARAM_LINKS>();
                                if (!links.isEmpty())
                                {
                                    const QModelIndex& linkIdx = links[0];
                                    oldLinks.append(UiHelper::exportLink(linkIdx));
                                }
                            }
                        }
                    }
                }
            }
            QMap<QString, NODE_DATA> newNodes;
            QList<EdgeInfo> newLinks;
            UiHelper::reAllocIdents(nodeCls, node.children, oldLinks, newNodes, newLinks);
            importNodes(newNodes, newLinks, QPointF(), this->index(nodeCls), true);
        }
    }
}
