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


GraphsModel::GraphsModel(QObject *parent)
    : IGraphsModel(parent)
    , m_selection(nullptr)
    , m_dirty(false)
    , m_linkModel(new LinkModel(this))
    , m_stack(new QUndoStack(this))
    , m_apiLevel(0)
    , m_bIOProcessing(false)
    , m_version(zenoio::VER_2_5)
{
    m_selection = new QItemSelectionModel(this);

    //link sync:
    connect(m_linkModel, &QAbstractItemModel::dataChanged, this, &GraphsModel::on_linkDataChanged);
    connect(m_linkModel, &QAbstractItemModel::rowsAboutToBeInserted, this, &GraphsModel::on_linkAboutToBeInserted);
    connect(m_linkModel, &QAbstractItemModel::rowsInserted, this, &GraphsModel::on_linkInserted);
    connect(m_linkModel, &QAbstractItemModel::rowsAboutToBeRemoved, this, &GraphsModel::on_linkAboutToBeRemoved);
    connect(m_linkModel, &QAbstractItemModel::rowsRemoved, this, &GraphsModel::on_linkRemoved);

    initDescriptors();
}

GraphsModel::~GraphsModel()
{
    clear();
}

QItemSelectionModel* GraphsModel::selectionModel() const
{
    return m_selection;
}

void GraphsModel::setFilePath(const QString& fn)
{
    m_filePath = fn;
    emit pathChanged(m_filePath);
}

SubGraphModel* GraphsModel::subGraph(const QString& name) const
{
    for (int i = 0; i < m_subGraphs.size(); i++)
    {
        if (m_subGraphs[i]->name() == name)
            return m_subGraphs[i];
    }
    return nullptr;
}

SubGraphModel* GraphsModel::subGraph(int idx) const
{
    if (idx >= 0 && idx < m_subGraphs.count())
    {
        return m_subGraphs[idx];
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
    if (graphName.compare("main", Qt::CaseInsensitive) == 0)
    {
        zeno::log_error("main graph is not allowed to be created or removed");
        return;
    }

    if (m_nodesDesc.find(graphName) != m_nodesDesc.end() ||
        m_subgsDesc.find(graphName) != m_subgsDesc.end())
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

    ZASSERT_EXIT(m_subgsDesc.find(oldName) != m_subgsDesc.end() &&
        m_subgsDesc.find(newName) == m_subgsDesc.end());

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

    NODE_DESC desc = m_subgsDesc[oldName];
    m_subgsDesc[newName] = desc;
    m_subgsDesc.remove(oldName);

    uint32_t ident = m_name2id[oldName];
    m_id2name[ident] = newName;
    ZASSERT_EXIT(m_name2id.find(oldName) != m_name2id.end());
    m_name2id.remove(oldName);
    m_name2id[newName] = ident;

    for (QString cate : desc.categories)
    {
        m_nodesCate[cate].nodes.removeAll(oldName);
        m_nodesCate[cate].nodes.append(newName);
    }

    emit graphRenamed(oldName, newName);
}

QModelIndex GraphsModel::nodeIndex(uint32_t id)
{
    for (int row = 0; row < m_subGraphs.size(); row++)
    {
        QModelIndex idx = m_subGraphs[row]->index(id);
        if (idx.isValid())
            return idx;
    }
    return QModelIndex();
}

QModelIndex GraphsModel::subgIndex(uint32_t sid)
{
    ZASSERT_EXIT(m_id2name.find(sid) != m_id2name.end(), QModelIndex());
    const QString& subgName = m_id2name[sid];
    return index(subgName);
}

QModelIndex GraphsModel::subgByNodeId(uint32_t id)
{
    for (int row = 0; row < m_subGraphs.size(); row++)
    {
        if (m_subGraphs[row]->index(id).isValid())
            return index(row, 0);
    }
    return QModelIndex();
}

QModelIndex GraphsModel::_createIndex(SubGraphModel* pSubModel) const
{
    if (!pSubModel)
        return QModelIndex();

    const QString& subgName = pSubModel->name();
    ZASSERT_EXIT(m_name2id.find(subgName) != m_name2id.end(), QModelIndex());
    int row = m_subGraphs.indexOf(pSubModel);
    return createIndex(row, 0, m_name2id[subgName]);
}

QModelIndex GraphsModel::index(int row, int column, const QModelIndex& parent) const
{
    if (row < 0 || row >= m_subGraphs.size())
        return QModelIndex();

    return _createIndex(m_subGraphs[row]);
}

QModelIndex GraphsModel::index(const QString& subGraphName) const
{
	for (int row = 0; row < m_subGraphs.size(); row++)
	{
		if (m_subGraphs[row]->name() == subGraphName)
		{
            return _createIndex(m_subGraphs[row]);
		}
	}
    return QModelIndex();
}

QModelIndex GraphsModel::indexBySubModel(SubGraphModel* pSubModel) const
{
    for (int row = 0; row < m_subGraphs.size(); row++)
    {
        if (m_subGraphs[row] == pSubModel)
            return _createIndex(pSubModel);
    }
    return QModelIndex();
}

QModelIndex GraphsModel::linkIndex(int r)
{
    return m_linkModel->index(r, 0);
}

QModelIndex GraphsModel::linkIndex(const QString& outNode,
                                   const QString& outSock,
                                   const QString& inNode,
                                   const QString& inSock)
{
    if (m_linkModel == nullptr)
        return QModelIndex();
    for (int r = 0; r < m_linkModel->rowCount(); r++)
    {
        QModelIndex idx = m_linkModel->index(r, 0);
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
    //format like: {subgraph-name}:{node-ident}:{[node|panel]param-layer-path}
    if (lst.size() == 1)
    {
        const QString& subgName = lst[0];
        return index(subgName);
    }
    else if (lst.size() == 2)
    {
        const QString& subgName = lst[0];
        const QString& nodeIdent = lst[1];
        const QModelIndex& subgIdx = index(subgName);
        return index(nodeIdent, subgIdx);
    }
    else if (lst.size() >= 3)
    {
        const QString& subgName = lst[0];
        const QString& nodeIdent = lst[1];
        QString paramPath = lst[2];
        const QModelIndex& subgIdx = index(subgName);
        const QModelIndex& nodeIdx = index(nodeIdent, subgIdx);
        if (!nodeIdx.isValid())
            return QModelIndex();
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
        case ROLE_OBJNAME:
            return m_subGraphs[index.row()]->name();
        case ROLE_OBJPATH:
        {
            const QString& subgName = m_subGraphs[index.row()]->name();
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
				QString type, name, defl;
				auto _arr = input.split('@');
				ZASSERT_EXIT(_arr.size() == 3, descs);
				type = _arr[0];
				name = _arr[1];
				defl = _arr[2];
				INPUT_SOCKET socket;
				socket.info.type = type;
				socket.info.name = name;
                CONTROL_INFO ctrlInfo = UiHelper::getControlByType(z_name, PARAM_INPUT, name, type);
                socket.info.control = ctrlInfo.control;
                socket.info.ctrlProps = ctrlInfo.controlProps.toMap();
				socket.info.defaultValue = UiHelper::parseStringByType(defl, type);
				desc.inputs[name] = socket;
			}
			for (QString output : outputs.split("%", QtSkipEmptyParts))
			{
				QString type, name, defl;
				auto _arr = output.split('@');
				ZASSERT_EXIT(_arr.size() == 3, descs);
				type = _arr[0];
				name = _arr[1];
				defl = _arr[2];
				OUTPUT_SOCKET socket;
				socket.info.type = type;
				socket.info.name = name;
                CONTROL_INFO ctrlInfo = UiHelper::getControlByType(z_name, PARAM_OUTPUT, name, type);
                socket.info.control = ctrlInfo.control;
                socket.info.ctrlProps = ctrlInfo.controlProps.toMap();
				socket.info.defaultValue = UiHelper::parseStringByType(defl, type);
				desc.outputs[name] = socket;
			}
			for (QString param : params.split("%", QtSkipEmptyParts))
			{
				QString type, name, defl;
				auto _arr = param.split('@');
				type = _arr[0];
				name = _arr[1];
				defl = _arr[2];

				PARAM_INFO paramInfo;
				paramInfo.bEnableConnect = false;
				paramInfo.name = name;
				paramInfo.typeDesc = type;
                CONTROL_INFO ctrlInfo = UiHelper::getControlByType(z_name, PARAM_PARAM, name, type);
                paramInfo.control = ctrlInfo.control;
                paramInfo.controlProps = ctrlInfo.controlProps;
				paramInfo.defaultValue = UiHelper::parseStringByType(defl, type);
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

void GraphsModel::initDescriptors()
{
    m_nodesDesc = getCoreDescs();
    m_nodesCate.clear();
    for (auto it = m_nodesDesc.constBegin(); it != m_nodesDesc.constEnd(); it++)
    {
        const QString& name = it.key();
        const NODE_DESC& desc = it.value();
        registerCate(desc);
    }

    //add Blackboard
    NODE_DESC desc;
    desc.name = "Blackboard";
    desc.categories.push_back("layout");
    m_nodesDesc.insert(desc.name, desc);
    registerCate(desc);

    //add Group
    NODE_DESC groupDesc;
    groupDesc.name = "Group";
    groupDesc.categories.push_back("layout");
    m_nodesDesc.insert(groupDesc.name, groupDesc);
    registerCate(groupDesc);
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

NODE_DESCS GraphsModel::descriptors() const
{
    NODE_DESCS descs;
    for (QString descName : m_subgsDesc.keys())
    {
        descs.insert(descName, m_subgsDesc[descName]);
    }
    for (QString nodeName : m_nodesDesc.keys())
    {
        //subgraph node has high priority than core node.
        if (descs.find(nodeName) == descs.end())
        {
            descs.insert(nodeName, m_nodesDesc[nodeName]);
        }
    }
    return descs;
}

bool GraphsModel::appendSubnetDescsFromZsg(const QList<NODE_DESC>& zsgSubnets)
{
    for (NODE_DESC desc : zsgSubnets)
    {
        if (m_subgsDesc.find(desc.name) == m_subgsDesc.end())
        {
            desc.is_subgraph = true;
            m_subgsDesc.insert(desc.name, desc);
            registerCate(desc);
        }
        else
        {
            zeno::log_error("The graph \"{}\" exists!", desc.name.toStdString());
            return false;
        }
    }
    return true;
}

void GraphsModel::registerCate(const NODE_DESC& desc)
{
    for (auto cate : desc.categories)
    {
        m_nodesCate[cate].name = cate;
        m_nodesCate[cate].nodes.push_back(desc.name);
    }
}

bool GraphsModel::getDescriptor(const QString& descName, NODE_DESC& desc)
{
    //internal node or subgraph node? if same name.
    if (m_subgsDesc.find(descName) != m_subgsDesc.end())
    {
        desc = m_subgsDesc[descName];
        return true;
    }
    if (m_nodesDesc.find(descName) != m_nodesDesc.end())
    {
        desc = m_nodesDesc[descName];
        return true;
    }
    return false;
}

bool GraphsModel::updateSubgDesc(const QString& descName, const NODE_DESC& desc)
{
    if (m_subgsDesc.find(descName) != m_subgsDesc.end())
    {
        m_subgsDesc[descName] = desc;
        return true;
    }
    return false;
}

void GraphsModel::appendSubGraph(SubGraphModel* pGraph)
{
    int row = m_subGraphs.size();
	beginInsertRows(QModelIndex(), row, row);
    m_subGraphs.append(pGraph);

    const QString& name = pGraph->name();
    QUuid uuid = QUuid::createUuid();
    uint32_t ident = uuid.data1;
    m_id2name[ident] = name;
    m_name2id[name] = ident;

	endInsertRows();
    //the subgraph desc has been inited when processing io.
    if (!IsIOProcessing())
    {
        NODE_DESC desc = getSubgraphDesc(pGraph);
        if (!desc.name.isEmpty() && m_subgsDesc.find(desc.name) == m_subgsDesc.end())
        {
            m_subgsDesc.insert(desc.name, desc);
            registerCate(desc);
        }
    }
}

void GraphsModel::removeGraph(int idx)
{
    beginRemoveRows(QModelIndex(), idx, idx);

    const QString& descName = m_subGraphs[idx]->name();
    m_subGraphs.remove(idx);

    ZASSERT_EXIT(m_name2id.find(descName) != m_name2id.end());
    uint32_t ident = m_name2id[descName];
    m_name2id.remove(descName);
    ZASSERT_EXIT(m_id2name.find(ident) != m_id2name.end());
    m_id2name.remove(ident);

    endRemoveRows();

    //if there is a core node shared the same name with this subgraph,
    // it will not be exported because it was omitted at begin.
    ZASSERT_EXIT(m_subgsDesc.find(descName) != m_subgsDesc.end());
    NODE_DESC desc = m_subgsDesc[descName];
    m_subgsDesc.remove(descName);
    for (QString cate : desc.categories)
    {
        m_nodesCate[cate].nodes.removeAll(descName);
    }
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

    QModelIndex newForkNodeIdx = pCurrentModel->index(subnetData[ROLE_OBJID].toString());
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
        NODE_DATA data;
        if (IsSubGraphNode(idx))
        {
            const QString& snodeId = idx.data(ROLE_OBJID).toString();
            const QString& ssubnetName = idx.data(ROLE_OBJNAME).toString();
            SubGraphModel* psSubModel = subGraph(ssubnetName);
            ZASSERT_EXIT(psSubModel, NODE_DATA());
            data = _fork(ssubnetName);
            const QString& subgNewNodeId = data[ROLE_OBJID].toString();

            nodes.insert(snodeId, pModel->nodeData(idx));
            oldGraphsToNew.insert(snodeId, data);
        }
        else
        {
            data = pModel->nodeData(idx);
            const QString &ident = idx.data(ROLE_OBJID).toString();
            nodes.insert(ident, data);
        }
    }
    for (int r = 0; r < m_linkModel->rowCount(); r++)
    {
        QModelIndex idx = m_linkModel->index(r, 0);
        const QString& outNode = idx.data(ROLE_OUTNODE).toString();
        const QString& inNode = idx.data(ROLE_INNODE).toString();
        if (nodes.find(inNode) != nodes.end() && nodes.find(outNode) != nodes.end())
        {
            QModelIndex outSockIdx = idx.data(ROLE_OUTSOCK_IDX).toModelIndex();
            QModelIndex inSockIdx = idx.data(ROLE_INSOCK_IDX).toModelIndex();
            const QString& outSockPath = outSockIdx.data(ROLE_OBJPATH).toString();
            const QString& inSockPath = inSockIdx.data(ROLE_OBJPATH).toString();
            links.append(EdgeInfo(outSockPath, inSockPath));
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
    NODE_DATA subnetData = NodesMgr::newNodeData(this, forkSubgName);
    subnetData[ROLE_OBJID] = UiHelper::generateUuid(forkName);
    subnetData[ROLE_OBJNAME] = forkName;
    //clear the link.
    OUTPUT_SOCKETS outputs = subnetData[ROLE_OUTPUTS].value<OUTPUT_SOCKETS>();
    for (auto it = outputs.begin(); it != outputs.end(); it++) {
        it->second.info.links.clear();
    }
    INPUT_SOCKETS inputs = subnetData[ROLE_INPUTS].value<INPUT_SOCKETS>();
    for (auto it = inputs.begin(); it != inputs.end(); it++) {
        it->second.info.links.clear();
    }
    subnetData[ROLE_INPUTS] = QVariant::fromValue(inputs);
    subnetData[ROLE_OUTPUTS] = QVariant::fromValue(outputs);
    //temp code: node pos.
    QPointF pos = subnetData[ROLE_OBJPOS].toPointF();
    pos.setY(pos.y() + 100);
    subnetData[ROLE_OBJPOS] = pos;
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

NODE_CATES GraphsModel::getCates()
{
    return m_nodesCate;
}

QString GraphsModel::filePath() const
{
    return m_filePath;
}

QString GraphsModel::fileName() const
{
    QFileInfo fi(m_filePath);
    QString fn;
    if (fi.isFile())
        fn = fi.fileName();
    return fn;
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
    if (IsIOProcessing())
        return;

    //todo: Thread safety
    m_apiLevel++;
}

void GraphsModel::endApiLevel()
{
    if (IsIOProcessing())
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
    for (int i = 0; i < m_subGraphs.size(); i++)
    {
        QModelIndex idx = m_subGraphs[i]->index(ident);
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
        QString id = nodeData[ROLE_OBJID].toString();
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
            addLink(link, false);
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
    if (IsIOProcessing() && m_subgsDesc.find(nodeName) != m_subgsDesc.end())
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

    ZASSERT_EXIT(inSockIdx.isValid() && outSockIdx.isValid());
    EdgeInfo link(outSockIdx.data(ROLE_OBJPATH).toString(), inSockIdx.data(ROLE_OBJPATH).toString());
    removeLink(link, enableTransaction);
}

void GraphsModel::removeLink(const EdgeInfo& link, bool enableTransaction)
{
    if (enableTransaction)
    {
        LinkCommand* pCmd = new LinkCommand(false, link, this);
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

        //restore the link
        QModelIndex linkIdx = m_linkModel->index(outSockIdx, inSockIdx);

        QAbstractItemModel* pOutputs = const_cast<QAbstractItemModel*>(outSockIdx.model());
        ZASSERT_EXIT(pOutputs);
        pOutputs->setData(outSockIdx, linkIdx, ROLE_REMOVELINK);

        QAbstractItemModel* pInputs = const_cast<QAbstractItemModel*>(inSockIdx.model());
        ZASSERT_EXIT(pInputs);
        pInputs->setData(inSockIdx, linkIdx, ROLE_REMOVELINK);

        ZASSERT_EXIT(linkIdx.isValid());
        m_linkModel->removeRow(linkIdx.row());
    }
}

QModelIndex GraphsModel::addLink(const QModelIndex& fromSock, const QModelIndex& toSock, bool enableTransaction)
{
    ZASSERT_EXIT(fromSock.isValid() && toSock.isValid(), QModelIndex());
    EdgeInfo link(fromSock.data(ROLE_OBJPATH).toString(), toSock.data(ROLE_OBJPATH).toString());
    return addLink(link, enableTransaction);
}

QModelIndex GraphsModel::addLink(const EdgeInfo& info, bool enableTransaction)
{
    if (enableTransaction)
    {
        LinkCommand* pCmd = new LinkCommand(true, info, this);
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
            zeno::log_warn("there is not valid input or output sockets.");
            return QModelIndex();
        }

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

    for (int i = 0; i < m_subGraphs.size(); i++)
    {
        if (m_subGraphs[i]->name() == name)
        {
            removeGraph(i);
        }
        else
        {
            m_subGraphs[i]->removeNodeByDescName(name);
        }
    }
}

QModelIndexList GraphsModel::findSubgraphNode(const QString& subgName)
{
    QModelIndexList nodes;
    for (int i = 0; i < m_subGraphs.size(); i++)
    {
        SubGraphModel* pModel = m_subGraphs[i];
        if (pModel->name() != subgName)
        {
            auto results = pModel->match(index(0, 0), ROLE_OBJNAME, subgName, -1, Qt::MatchExactly);
            nodes.append(results);
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
    m_linkModel->clear();
    emit modelClear();
}

QModelIndexList GraphsModel::searchInSubgraph(const QString& objName, const QModelIndex& subgIdx)
{
    SubGraphModel* pModel = subGraph(subgIdx.row());
    QModelIndexList list;
    auto count = pModel->rowCount();

    for (auto i = 0; i < count; i++) {
        auto index = pModel->index(i, 0);
        auto item = pModel->nodeData(index);
        if (item[ROLE_OBJID].toString().contains(objName, Qt::CaseInsensitive)) {
            list.append(index);
        }
        else {
            QString _type("string");
            bool inserted = false;
            {
                auto params = item[ROLE_PARAMETERS].value<PARAMS_INFO>();
                auto iter = params.begin();
                while (iter != params.end()) {
                    if (iter.value().typeDesc == _type) {
                        if (iter.value().value.toString().contains(objName, Qt::CaseInsensitive)) {
                            list.append(index);
                            inserted = true;
                            break;
                        }
                    }
                    ++iter;
                }
            }
            if (inserted) {
                continue;
            }
            {
                auto inputs = item[ROLE_INPUTS].value<INPUT_SOCKETS>();
                auto iter = inputs.begin();
                while (iter != inputs.end()) {
                    if (iter->value().info.type == _type) {
                        if (iter->value().info.defaultValue.toString().contains(objName, Qt::CaseInsensitive)) {
                            list.append(index);
                            inserted = true;
                            break;
                        }
                    }
                    ++iter;
                }

            }
        }
    }
    return list;
}

QModelIndexList GraphsModel::subgraphsIndice() const
{
    return persistentIndexList();
}

LinkModel* GraphsModel::linkModel() const
{
    return m_linkModel;
}

void GraphsModel::on_subg_dataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles)
{
    SubGraphModel* pSubModel = qobject_cast<SubGraphModel*>(sender());

    ZASSERT_EXIT(pSubModel && roles.size() == 1);
    QModelIndex subgIdx = indexBySubModel(pSubModel);
    emit _dataChanged(subgIdx, topLeft, roles[0]);
}

void GraphsModel::on_subg_rowsAboutToBeInserted(const QModelIndex& parent, int first, int last)
{
    SubGraphModel* pSubModel = qobject_cast<SubGraphModel*>(sender());

    ZASSERT_EXIT(pSubModel);
    QModelIndex subgIdx = indexBySubModel(pSubModel);
    emit _rowsAboutToBeInserted(subgIdx, first, last);
}

void GraphsModel::on_subg_rowsInserted(const QModelIndex& parent, int first, int last)
{
    SubGraphModel* pSubModel = qobject_cast<SubGraphModel*>(sender());
    ZASSERT_EXIT(pSubModel);
    QModelIndex subgIdx = indexBySubModel(pSubModel);
    emit _rowsInserted(subgIdx, parent, first, last);
}

void GraphsModel::on_subg_rowsAboutToBeRemoved(const QModelIndex& parent, int first, int last)
{
    SubGraphModel* pSubModel = qobject_cast<SubGraphModel*>(sender());
    ZASSERT_EXIT(pSubModel);
    QModelIndex subgIdx = indexBySubModel(pSubModel);
    emit _rowsAboutToBeRemoved(subgIdx, parent, first, last);
}

void GraphsModel::on_subg_rowsRemoved(const QModelIndex& parent, int first, int last)
{
    SubGraphModel* pSubModel = qobject_cast<SubGraphModel*>(sender());

    ZASSERT_EXIT(pSubModel);
    QModelIndex subgIdx = indexBySubModel(pSubModel);
    emit _rowsRemoved(parent, first, last);
}

QModelIndex GraphsModel::getSubgraphIndex(const QModelIndex& linkIdx)
{
	const QString& inNode = linkIdx.data(ROLE_INNODE).toString();
    for (int r = 0; r < m_subGraphs.size(); r++)
    {
        SubGraphModel* pSubModel = m_subGraphs[r];
		if (pSubModel->index(inNode).isValid())
		{
            return index(r, 0);
		}
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
     QModelIndex linkIdx = m_linkModel->index(first, 0, parent);
     const QModelIndex& subgIdx = getSubgraphIndex(linkIdx);
     if (subgIdx.isValid())
        emit linkAboutToBeInserted(subgIdx, parent, first, last);
}

void GraphsModel::on_linkInserted(const QModelIndex& parent, int first, int last)
{
	QModelIndex linkIdx = m_linkModel->index(first, 0, parent);
	const QModelIndex& subgIdx = getSubgraphIndex(linkIdx);
    if (subgIdx.isValid())
	    emit linkInserted(subgIdx, parent, first, last);
}

void GraphsModel::on_linkAboutToBeRemoved(const QModelIndex& parent, int first, int last)
{
	QModelIndex linkIdx = m_linkModel->index(first, 0, parent);
	const QModelIndex& subgIdx = getSubgraphIndex(linkIdx);
    if (subgIdx.isValid())
	    emit linkAboutToBeRemoved(subgIdx, parent, first, last);
}

void GraphsModel::on_linkRemoved(const QModelIndex& parent, int first, int last)
{
	QModelIndex linkIdx = m_linkModel->index(first, 0, parent);
	const QModelIndex& subgIdx = getSubgraphIndex(linkIdx);
    if (subgIdx.isValid())
	    emit linkRemoved(subgIdx, parent, first, last);
}

bool GraphsModel::onSubIOAdd(SubGraphModel* pGraph, NODE_DATA nodeData2)
{
    const QString& descName = nodeData2[ROLE_OBJNAME].toString();
    if (descName != "SubInput" && descName != "SubOutput")
        return false;

    bool bInput = descName == "SubInput";

    PARAMS_INFO params = nodeData2[ROLE_PARAMETERS].value<PARAMS_INFO>();
    ZASSERT_EXIT(params.find("name") != params.end(), false);
    PARAM_INFO& param = params["name"];
    QString newSockName = UiHelper::correctSubIOName(this, pGraph->name(), param.value.toString(), bInput);
    param.value = newSockName;
    nodeData2[ROLE_PARAMETERS] = QVariant::fromValue(params);
    pGraph->appendItem(nodeData2);

    if (!IsIOProcessing())
    {
        const QModelIndex& nodeIdx = pGraph->index(nodeData2[ROLE_OBJID].toString());
        onSubIOAddRemove(pGraph, nodeIdx, bInput, true);
    }
    return true;
}

bool GraphsModel::onListDictAdd(SubGraphModel* pGraph, NODE_DATA nodeData2)
{
    const QString& descName = nodeData2[ROLE_OBJNAME].toString();
    if (descName == "MakeList" || descName == "MakeDict")
    {
        INPUT_SOCKETS inputs = nodeData2[ROLE_INPUTS].value<INPUT_SOCKETS>();
        INPUT_SOCKET inSocket;
        inSocket.info.nodeid = nodeData2[ROLE_OBJID].toString();

        int maxObjId = UiHelper::getMaxObjId(inputs.keys());
        if (maxObjId == -1)
        {
            inSocket.info.name = "obj0";
            if (descName == "MakeDict")
            {
                inSocket.info.control = CONTROL_NONE;
                inSocket.info.sockProp = SOCKPROP_EDITABLE;
            }
            inputs.insert(inSocket.info.name, inSocket);
            nodeData2[ROLE_INPUTS] = QVariant::fromValue(inputs);
        }
        pGraph->appendItem(nodeData2);
        return true;
    }
    else if (descName == "ExtractDict")
    {
        OUTPUT_SOCKETS outputs = nodeData2[ROLE_OUTPUTS].value<OUTPUT_SOCKETS>();
        OUTPUT_SOCKET outSocket;
        outSocket.info.nodeid = nodeData2[ROLE_OBJID].toString();

        int maxObjId = UiHelper::getMaxObjId(outputs.keys());
        if (maxObjId == -1)
        {
            outSocket.info.name = "obj0";
            outSocket.info.control = CONTROL_NONE;
            outSocket.info.sockProp = SOCKPROP_EDITABLE;
            outputs.insert(outSocket.info.name, outSocket);
            nodeData2[ROLE_OUTPUTS] = QVariant::fromValue(outputs);
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

    ZASSERT_EXIT(m_subgsDesc.find(subnetNodeName) != m_subgsDesc.end());
    NODE_DESC& desc = m_subgsDesc[subnetNodeName];

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
            ZASSERT_EXIT(desc.inputs.find(nameValue) == desc.inputs.end());
            desc.inputs[nameValue].info = info;
        }
        else
        {
            ZASSERT_EXIT(desc.outputs.find(nameValue) == desc.outputs.end());
            desc.outputs[nameValue].info = info;
        }

        //sync to all subgraph nodes.
        QModelIndexList subgNodes = findSubgraphNode(subnetNodeName);
        for (QModelIndex subgNode : subgNodes)
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
        for (QModelIndex subgNode : subgNodes)
        {
            NodeParamModel* nodeParams = QVariantPtr<NodeParamModel>::asPtr(subgNode.data(ROLE_NODE_PARAMS));
            nodeParams->removeParam(bInput ? PARAM_INPUT : PARAM_OUTPUT, nameValue);
        }
    }
}

QList<SEARCH_RESULT> GraphsModel::search(const QString& content, int searchOpts)
{
    QList<SEARCH_RESULT> results;
    if (content.isEmpty())
        return results;

    QSet<QString> nodes;
    if (searchOpts & SEARCH_SUBNET)
    {
        QModelIndexList lst = match(index(0, 0), ROLE_OBJNAME, content, -1, Qt::MatchContains);
        for (QModelIndex subgIdx : lst)
        {
            SEARCH_RESULT result;
            result.targetIdx = subgIdx;
            result.type = SEARCH_SUBNET;
            results.append(result);
        }
    }
    /* start match len*/
    static int sStartMatch = 2;
    if ((searchOpts & SEARCH_ARGS) && content.size() >= sStartMatch)
    {
        for (auto subgInfo : m_subGraphs)
        {
            SubGraphModel* pModel = subgInfo;
            QModelIndex subgIdx = indexBySubModel(pModel);
            for (int r = 0; r < pModel->rowCount(); r++)
            {
                QModelIndex nodeIdx = pModel->index(r, 0);
                INPUT_SOCKETS inputs = nodeIdx.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
                PARAMS_INFO params = nodeIdx.data(ROLE_PARAMETERS).value<PARAMS_INFO>();
                for (INPUT_SOCKET inputSock : inputs)
                {
                    if (inputSock.info.type == "string" ||
                        inputSock.info.type == "multiline_string" ||
                        inputSock.info.type == "readpath" ||
                        inputSock.info.type == "writepath")
                    {
                        const QString& textValue = inputSock.info.defaultValue.toString();
                        if (textValue.contains(content, Qt::CaseInsensitive))
                        {
                SEARCH_RESULT result;
                result.targetIdx = nodeIdx;
                result.subgIdx = subgIdx;
                            result.type = SEARCH_ARGS;
                            result.socket = inputSock.info.name;
                results.append(result);
            }
        }
    }
                for (PARAM_INFO param : params)
                {
                    if (param.typeDesc == "string" ||
                        param.typeDesc == "multiline_string" ||
                        param.typeDesc == "readpath" ||
                        param.typeDesc == "writepath")
                    {
                        const QString& textValue = param.value.toString();
                        if (textValue.contains(content, Qt::CaseInsensitive))
                        {
                            SEARCH_RESULT result;
                            result.targetIdx = nodeIdx;
                            result.subgIdx = subgIdx;
                            result.type = SEARCH_ARGS;
                            result.socket = param.name;
                            results.append(result);
                        }
                    }
                }
            }
        }
    }
    if (searchOpts & SEARCH_NODEID)
    {
        for (auto subgInfo : m_subGraphs)
        {
            SubGraphModel* pModel = subgInfo;
            QModelIndex subgIdx = indexBySubModel(pModel);
            QModelIndexList lst = pModel->match(pModel->index(0, 0), ROLE_OBJID, content, -1, Qt::MatchContains);
            if (!lst.isEmpty())
            {
                for (QModelIndex nodeIdx : lst)
                {
                    SEARCH_RESULT result;
                    result.targetIdx = nodeIdx;
                    result.subgIdx = subgIdx;
                    result.type = SEARCH_NODEID;
                    results.append(result);
                    nodes.insert(nodeIdx.data(ROLE_OBJID).toString());
                }
            }
        }
    }
    if (searchOpts & SEARCH_NODECLS)
    {
        for (auto subgInfo : m_subGraphs)
        {
            SubGraphModel* pModel = subgInfo;
            QModelIndex subgIdx = indexBySubModel(pModel);
            //todo: searching by key.
            QModelIndexList lst = pModel->match(pModel->index(0, 0), ROLE_OBJNAME, content, -1, Qt::MatchContains);
            for (QModelIndex nodeIdx : lst)
            {
                if (nodes.contains(nodeIdx.data(ROLE_OBJID).toString()))
                {
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
    }
    if (searchOpts & SEARCH_NODECLS)
    {
        for (auto subgInfo : m_subGraphs)
        {
            SubGraphModel* pModel = subgInfo;
            QModelIndex subgIdx = indexBySubModel(pModel);
            //todo: searching by key.
            QModelIndexList lst = pModel->match(pModel->index(0, 0), ROLE_OBJNAME, content, -1, Qt::MatchContains);
            for (QModelIndex nodeIdx : lst)
            {
                if (nodes.contains(nodeIdx.data(ROLE_OBJID).toString()))
                {
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

bool GraphsModel::hasDescriptor(const QString& nodeName) const
{
    return m_nodesDesc.find(nodeName) != m_nodesDesc.end() ||
        m_subgsDesc.find(nodeName) != m_subgsDesc.end();
}

void GraphsModel::setNodeData(const QModelIndex &nodeIndex, const QModelIndex &subGpIdx, const QVariant &value, int role) {
    SubGraphModel* pModel = this->subGraph(subGpIdx.row());
    ZASSERT_EXIT(pModel);
    pModel->setData(nodeIndex, value, role);
}
