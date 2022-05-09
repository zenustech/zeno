#include "subgraphmodel.h"
#include "graphsmodel.h"
#include <zenoui/model/modelrole.h>
#include <zenoui/util/uihelper.h>
#include "util/apphelper.h"
#include "util/log.h"
#include <zeno/zeno.h>
#include <zenoui/util/cihou.h>
#include "zenoapplication.h"


GraphsModel::GraphsModel(QObject *parent)
    : IGraphsModel(parent)
    , m_selection(nullptr)
    , m_dirty(false)
    , m_linkModel(new QStandardItemModel(this))
    , m_stack(new QUndoStack(this))
{
    m_selection = new QItemSelectionModel(this);

    //link sync:
    connect(m_linkModel, &QAbstractItemModel::dataChanged, this, &GraphsModel::on_linkDataChanged);
    connect(m_linkModel, &QAbstractItemModel::rowsAboutToBeInserted, this, &GraphsModel::on_linkAboutToBeInserted);
    connect(m_linkModel, &QAbstractItemModel::rowsInserted, this, &GraphsModel::on_linkInserted);
    connect(m_linkModel, &QAbstractItemModel::rowsAboutToBeRemoved, this, &GraphsModel::on_linkAboutToBeRemoved);
    connect(m_linkModel, &QAbstractItemModel::rowsRemoved, this, &GraphsModel::on_linkRemoved);
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
        if (m_subGraphs[i].pModel->name() == name)
            return m_subGraphs[i].pModel;
    }
    return nullptr;
}

SubGraphModel* GraphsModel::subGraph(int idx) const
{
    if (idx >= 0 && idx < m_subGraphs.count())
    {
        return m_subGraphs[idx].pModel;
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

void GraphsModel::newSubgraph(const QString &graphName)
{
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

void GraphsModel::reloadSubGraph(const QString& graphName)
{
    initDescriptors();
    SubGraphModel *pReloadModel = subGraph(graphName);
    ZASSERT_EXIT(pReloadModel);
    NODES_DATA datas = pReloadModel->nodes();

    pReloadModel->clear();

    pReloadModel->blockSignals(true);
    for (auto data : datas)
    {
        pReloadModel->appendItem(data);
    }
    pReloadModel->blockSignals(false);
    pReloadModel->reload();
    markDirty();
}

void GraphsModel::renameSubGraph(const QString& oldName, const QString& newName)
{
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
    initDescriptors();
    emit graphRenamed(oldName, newName);
}

QModelIndex GraphsModel::index(int row, int column, const QModelIndex& parent) const
{
    if (row < 0 || row >= m_subGraphs.size())
        return QModelIndex();

    return createIndex(row, 0, nullptr);
}

QModelIndex GraphsModel::index(const QString& subGraphName) const
{
	for (int row = 0; row < m_subGraphs.size(); row++)
	{
		if (m_subGraphs[row].pModel->name() == subGraphName)
		{
            return createIndex(row, 0, nullptr);
		}
	}
    return QModelIndex();
}

QModelIndex GraphsModel::indexBySubModel(SubGraphModel* pSubModel) const
{
    for (int row = 0; row < m_subGraphs.size(); row++)
    {
        if (m_subGraphs[row].pModel == pSubModel)
            return createIndex(row, 0, nullptr);
    }
    return QModelIndex();
}

QModelIndex GraphsModel::linkIndex(int r)
{
    return m_linkModel->index(r, 0);
}

QModelIndex GraphsModel::parent(const QModelIndex& child) const
{
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
        return m_subGraphs[index.row()].pModel->name();
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
			else
			{
                //new subgraph.
                pModel->setName(newName);
                initDescriptors();
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
                socket.info.control = UiHelper::_getControlType(type);
				socket.info.defaultValue = UiHelper::_parseDefaultValue(defl, type);
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
                socket.info.control = UiHelper::_getControlType(type);
				socket.info.defaultValue = UiHelper::_parseDefaultValue(defl, type);
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
				paramInfo.control = UiHelper::_getControlType(type);
				paramInfo.defaultValue = UiHelper::_parseDefaultValue(defl, type);
				//thers is no "value" in descriptor, but it's convient to initialize param value. 
				paramInfo.value = paramInfo.defaultValue;
				desc.params[name] = paramInfo;
			}
			desc.categories = z_categories;

			descs.insert(z_name, desc);
		}
	}
    return descs;
}

void GraphsModel::initDescriptors()
{
    NODE_DESCS descs = getCoreDescs();
    NODE_DESCS subgDescs = getSubgraphDescs();
    for (QString key : subgDescs.keys())
    {
        ZASSERT_EXIT(descs.find(key) == descs.end());
        descs.insert(key, subgDescs[key]);
    }
    setDescriptors(descs);
}

NODE_DESCS GraphsModel::getSubgraphDescs()
{
    NODE_DESCS descs;
    for (int r = 0; r < this->rowCount(); r++)
    {
        QModelIndex index = this->index(r, 0);
        ZASSERT_EXIT(index.isValid(), descs);
        SubGraphModel *pModel = subGraph(r);
        const QString& graphName = pModel->name();
        if (graphName == "main" || graphName.isEmpty())
            continue;

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
        descs[graphName] = desc;
    }
    return descs;
}

NODE_DESCS GraphsModel::descriptors() const
{
    return m_nodesDesc;
}

void GraphsModel::setDescriptors(const NODE_DESCS& nodeDescs)
{
    m_nodesDesc = nodeDescs;
    m_nodesCate.clear();
    for (auto it = m_nodesDesc.constBegin(); it != m_nodesDesc.constEnd(); it++)
    {
        const QString& name = it.key();
        const NODE_DESC& desc = it.value();
        for (auto cate : desc.categories)
        {
            m_nodesCate[cate].name = cate;
            m_nodesCate[cate].nodes.push_back(name);
        }
    }
}

void GraphsModel::appendSubGraph(SubGraphModel* pGraph)
{
    int row = m_subGraphs.size();
	beginInsertRows(QModelIndex(), row, row);
    SUBMODEL_SCENE info;
    info.pModel = pGraph;
    m_subGraphs.append(info);
    initDescriptors();
	endInsertRows();
}

void GraphsModel::removeGraph(int idx)
{
    beginRemoveRows(QModelIndex(), idx, idx);
    m_subGraphs.remove(idx);
    endRemoveRows();
    markDirty();
}

QModelIndex GraphsModel::fork(const QModelIndex& whichSubg, const QModelIndex& subnetNodeIdx)
{
    const QString& subnetName = subnetNodeIdx.data(ROLE_OBJNAME).toString();
    SubGraphModel* pModel = subGraph(subnetName);
    ZASSERT_EXIT(pModel, QModelIndex());

	SubGraphModel* pForkModel = new SubGraphModel(this);
    const QString& forkName = subnetName + " (copy)";
    pForkModel->setName(forkName);
    appendSubGraph(pForkModel);

    QModelIndex subgIdx = indexBySubModel(pModel);

    QModelIndexList nodeIndice, linkIndice;
    for (int r = 0; r < pModel->rowCount(); r++)
    {
        QModelIndex idx = pModel->index(r, 0);
        nodeIndice.append(idx);

        const INPUT_SOCKETS& inputs = idx.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
        for (auto input : inputs.values())
        {
            for (auto linkIdx : input.linkIndice)
            {
                linkIndice.append(linkIdx);
            }
        }
        const OUTPUT_SOCKETS& outputs = idx.data(ROLE_OUTPUTS).value<OUTPUT_SOCKETS>();
		for (auto output : outputs.values())
		{
			for (auto linkIdx : output.linkIndice)
			{
				linkIndice.append(linkIdx);
			}
		}
    }

    QMap<QString, NODE_DATA> items = UiHelper::dumpItems(this, subgIdx, nodeIndice, linkIndice);
    QList<NODE_DATA> datas;
    for (NODE_DATA data : items)
    {
        datas.append(data);
    }

    QModelIndex newSubgIdx = indexBySubModel(pForkModel);
    importNodeLinks(datas, newSubgIdx);

    //create the new fork subnet node at graph indexed by whichSubg.
    NODE_DATA subnetData = itemData(subnetNodeIdx, whichSubg);
    subnetData[ROLE_OBJID] = UiHelper::generateUuid(forkName);
    subnetData[ROLE_OBJNAME] = forkName;

    OUTPUT_SOCKETS outputs = subnetData[ROLE_OUTPUTS].value<OUTPUT_SOCKETS>();
	for (auto it = outputs.begin(); it != outputs.end(); it++)
	{
		it->second.linkIndice.clear();
		it->second.inNodes.clear();
	}
    INPUT_SOCKETS inputs = subnetData[ROLE_INPUTS].value<INPUT_SOCKETS>();
	for (auto it = inputs.begin(); it != inputs.end(); it++)
	{
		it->second.linkIndice.clear();
		it->second.outNodes.clear();
	}
    subnetData[ROLE_INPUTS] = QVariant::fromValue(inputs);
    subnetData[ROLE_OUTPUTS] = QVariant::fromValue(outputs);

    //temp code: node pos.
    QPointF pos = subnetData[ROLE_OBJPOS].toPointF();
    pos.setY(pos.y() + 100);
    subnetData[ROLE_OBJPOS] = pos;

    SubGraphModel* pCurrentModel = subGraph(whichSubg.row());
    pCurrentModel->appendItem(subnetData, false);

    return newSubgIdx;
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
}

void GraphsModel::endTransaction()
{
    m_stack->endMacro();
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

QVariant GraphsModel::data2(const QModelIndex& subGpIdx, const QModelIndex& index, int role)
{
	SubGraphModel* pGraph = subGraph(subGpIdx.row());
    ZASSERT_EXIT(pGraph, QVariant());
    return pGraph->data(index, role);
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
    bool bEnableIOProc = zenoApp->IsIOProcessing();
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
        SubGraphModel* pGraph = subGraph(subGpIdx.row());
        ZASSERT_EXIT(pGraph);

        NODE_DATA nodeData2 = nodeData;
        PARAMS_INFO params = nodeData2[ROLE_PARAMETERS].value<PARAMS_INFO>();
        QString descName = nodeData[ROLE_OBJNAME].toString();
        if (descName == "SubInput" || descName == "SubOutput")
        {
            AppHelper::correctSubIOName(this, subGpIdx, descName, params);
            nodeData2[ROLE_PARAMETERS] = QVariant::fromValue(params);
            pGraph->appendItem(nodeData2);
        }
        else
        {
            pGraph->appendItem(nodeData);
        }

        //update desc if meet subinput/suboutput node.
        if (!bEnableIOProc)
        {
            const QModelIndex &idx = pGraph->index(nodeData[ROLE_OBJID].toString());
            const QString &objName = idx.data(ROLE_OBJNAME).toString();
            bool bInserted = true;
            if (objName == "SubInput")
                onSubInfoChanged(pGraph, idx, true, bInserted);
            else if (objName == "SubOutput") 
                onSubInfoChanged(pGraph, idx, false, bInserted);
        }
    }
}

void GraphsModel::removeNode(const QString& nodeid, const QModelIndex& subGpIdx, bool enableTransaction)
{
	SubGraphModel* pGraph = subGraph(subGpIdx.row());
    ZASSERT_EXIT(pGraph);

    bool bEnableIOProc = zenoApp->IsIOProcessing();
    if (bEnableIOProc)
        enableTransaction = false;

    if (enableTransaction)
    {
        QModelIndex idx = pGraph->index(nodeid);
        int row = idx.row();
        const NODE_DATA& data = pGraph->itemData(idx);

        RemoveNodeCommand* pCmd = new RemoveNodeCommand(row, data, this, subGpIdx);
        m_stack->push(pCmd);
    }
    else
    {
        
        const QModelIndex &idx = pGraph->index(nodeid);
        const QString &objName = idx.data(ROLE_OBJNAME).toString();
        if (!bEnableIOProc)
        {
            //if subnode removed, the whole graphs referred to it should be update.
            bool bInserted = false;
            if (objName == "SubInput")
            {
                onSubInfoChanged(pGraph, idx, true, bInserted);
            }
            else if (objName == "SubOutput")
            {
                onSubInfoChanged(pGraph, idx, false, bInserted);
            }
        }
        pGraph->removeNode(nodeid, false);
    }
}

void GraphsModel::insertRow(int row, const NODE_DATA& nodeData, const QModelIndex& subGpIdx)
{
    //only implementation, no transaction.
	SubGraphModel* pGraph = subGraph(subGpIdx.row());
    ZASSERT_EXIT(pGraph);
    pGraph->insertRow(row, nodeData);
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

void GraphsModel::importNodeLinks(const QList<NODE_DATA>& nodes, const QModelIndex& subGpIdx)
{
	beginTransaction("import nodes");

	appendNodes(nodes, subGpIdx, true);
	//add links for pasted node.
	for (int i = 0; i < nodes.size(); i++)
	{
		const NODE_DATA& data = nodes[i];
		const QString& inNode = data[ROLE_OBJID].toString();
		INPUT_SOCKETS inputs = data[ROLE_INPUTS].value<INPUT_SOCKETS>();
		foreach(const QString & inSockName, inputs.keys())
		{
			const INPUT_SOCKET& inSocket = inputs[inSockName];
			for (const QString& outNode : inSocket.outNodes.keys())
			{
				for (const QString& outSock : inSocket.outNodes[outNode].keys())
				{
					const QModelIndex& outIdx = index(outNode, subGpIdx);
					if (outIdx.isValid())
					{
						addLink(EdgeInfo(outNode, inNode, outSock, inSockName), subGpIdx, true);
					}
				}
			}
		}
	}
    endTransaction();
}

void GraphsModel::copyPaste(const QModelIndex &fromSubg, const QModelIndexList &srcNodes, const QModelIndex &toSubg, QPointF pos, bool enableTrans)
{
    if (!fromSubg.isValid() || srcNodes.isEmpty() || !toSubg.isValid())
        return;

    if (enableTrans)
        beginTransaction("copy paste");

    SubGraphModel* srcGraph = subGraph(fromSubg.row());
    ZASSERT_EXIT(srcGraph);

    SubGraphModel* dstGraph = subGraph(toSubg.row());
    ZASSERT_EXIT(dstGraph);

    QMap<QString, QString> old2New, new2old;

    QMap<QString, NODE_DATA> oldNodes;
    for (QModelIndex idx : srcNodes)
    {
        NODE_DATA old = srcGraph->itemData(idx);
        oldNodes.insert(old[ROLE_OBJID].toString(), old);
    }
    QPointF offset = pos - (*oldNodes.begin())[ROLE_OBJPOS].toPointF();

    QMap<QString, NODE_DATA> newNodes;
    for (NODE_DATA old : oldNodes)
    {
        NODE_DATA newNode = old;
        const INPUT_SOCKETS inputs = newNode[ROLE_INPUTS].value<INPUT_SOCKETS>();
        INPUT_SOCKETS newInputs = inputs;
        const OUTPUT_SOCKETS outputs = newNode[ROLE_OUTPUTS].value<OUTPUT_SOCKETS>();
        OUTPUT_SOCKETS newOutputs = outputs;

        for (INPUT_SOCKET& inSocket : newInputs)
        {
            inSocket.linkIndice.clear();
            inSocket.outNodes.clear();
        }
        newNode[ROLE_INPUTS] = QVariant::fromValue(newInputs);

        for (OUTPUT_SOCKET& outSocket : newOutputs)
        {
            outSocket.linkIndice.clear();
            outSocket.inNodes.clear();
        }
        newNode[ROLE_OUTPUTS] = QVariant::fromValue(newOutputs);

        QString nodeName = old[ROLE_OBJNAME].toString();
        const QString& oldId = old[ROLE_OBJID].toString();
        const QString& newId = UiHelper::generateUuid(nodeName);

        newNode[ROLE_OBJPOS] = old[ROLE_OBJPOS].toPointF() + offset;
        newNode[ROLE_OBJID] = newId;

        newNodes.insert(newId, newNode);

        old2New.insert(oldId, newId);
        new2old.insert(newId, oldId);
    }

    QList<NODE_DATA> lstNodes;
    for (NODE_DATA data : newNodes)
        lstNodes.append(data);
    appendNodes(lstNodes, toSubg, enableTrans);

    //reconstruct topology for new node.
    for (NODE_DATA newNode : newNodes)
    {
        const QString& newId = newNode[ROLE_OBJID].toString();
        const QString& oldId = new2old[newId];

        const NODE_DATA& oldData = oldNodes[oldId];

        const INPUT_SOCKETS &oldInputs = oldData[ROLE_INPUTS].value<INPUT_SOCKETS>();
        const OUTPUT_SOCKETS &oldOutputs = oldData[ROLE_OUTPUTS].value<OUTPUT_SOCKETS>();

        INPUT_SOCKETS inputs = newNode[ROLE_INPUTS].value<INPUT_SOCKETS>();
        OUTPUT_SOCKETS outputs = newNode[ROLE_OUTPUTS].value<OUTPUT_SOCKETS>();

        for (INPUT_SOCKET inSock : oldInputs)
        {
            for (QPersistentModelIndex linkIdx : inSock.linkIndice)
            {
                QString inNode = linkIdx.data(ROLE_INNODE).toString();
                QString inSock = linkIdx.data(ROLE_INSOCK).toString();
                QString outNode = linkIdx.data(ROLE_OUTNODE).toString();
                QString outSock = linkIdx.data(ROLE_OUTSOCK).toString();

                if (oldNodes.find(inNode) != oldNodes.end() && oldNodes.find(outNode) != oldNodes.end())
                {
                    QString newOutNode, newInNode;
                    newOutNode = old2New[outNode];
                    newInNode = old2New[inNode];
                    addLink(EdgeInfo(newOutNode, newInNode, outSock, inSock), toSubg, enableTrans);
                }
            }
        }
    }
    if (enableTrans)
        endTransaction();
}

QModelIndex GraphsModel::extractSubGraph(const QModelIndexList& nodes, const QModelIndex& fromSubgIdx, const QString& toSubg, bool enableTrans)
{
    if (nodes.isEmpty() || !fromSubgIdx.isValid() || toSubg.isEmpty() || subGraph(toSubg))
    {
        return QModelIndex();
    }

    ZASSERT_EXIT(false, QModelIndex());

    enableTrans = true;    //dangerous to trans...
    if (enableTrans)
        beginTransaction("extract a new graph");

    //first, new the target subgraph
    newSubgraph(toSubg);
    QModelIndex toSubgIdx = index(toSubg);

    //copy nodes to new subg.
    copyPaste(fromSubgIdx, nodes, toSubgIdx, QPointF(0, 0), enableTrans);

    //remove nodes from old subg.
    QStringList ids;
    for (QModelIndex idx : nodes)
        ids.push_back(idx.data(ROLE_OBJID).toString());
    for (QString id : ids)
        removeNode(id, fromSubgIdx, enableTrans);

    if (enableTrans)
        endTransaction();

    return toSubgIdx;
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


void GraphsModel::removeLinks(const QList<QPersistentModelIndex>& info, const QModelIndex& subGpIdx, bool enableTransaction)
{
    for (const QPersistentModelIndex& linkIdx : info)
    {
        removeLink(linkIdx, subGpIdx, enableTransaction);
    }
}

void GraphsModel::removeLink(const QPersistentModelIndex& linkIdx, const QModelIndex& subGpIdx, bool enableTransaction)
{
    if (!linkIdx.isValid())
        return;

    if (enableTransaction)
    {
		RemoveLinkCommand* pCmd = new RemoveLinkCommand(linkIdx, this, subGpIdx);
		m_stack->push(pCmd);
    }
    else
    {
		SubGraphModel* pGraph = subGraph(subGpIdx.row());
        ZASSERT_EXIT(pGraph && linkIdx.isValid());
		if (pGraph)
		{
			const QString& outNode = linkIdx.data(ROLE_OUTNODE).toString();
			const QString& outSock = linkIdx.data(ROLE_OUTSOCK).toString();
			const QString& inNode = linkIdx.data(ROLE_INNODE).toString();
			const QString& inSock = linkIdx.data(ROLE_INSOCK).toString();

			const QModelIndex& outIdx = pGraph->index(outNode);
			const QModelIndex& inIdx = pGraph->index(inNode);

			OUTPUT_SOCKETS outputs = pGraph->data(outIdx, ROLE_OUTPUTS).value<OUTPUT_SOCKETS>();
            if (outputs.find(outSock) != outputs.end())
            {
				outputs[outSock].linkIndice.removeOne(linkIdx);
				pGraph->setData(outIdx, QVariant::fromValue(outputs), ROLE_OUTPUTS);
            }

			INPUT_SOCKETS inputs = pGraph->data(inIdx, ROLE_INPUTS).value<INPUT_SOCKETS>();
            if (inputs.find(inSock) != inputs.end())
            {
				inputs[inSock].linkIndice.removeOne(linkIdx);
				pGraph->setData(inIdx, QVariant::fromValue(inputs), ROLE_INPUTS);
            }
		}
		m_linkModel->removeRow(linkIdx.row());
    }
}

QModelIndex GraphsModel::addLink(const EdgeInfo& info, const QModelIndex& subGpIdx, bool enableTransaction)
{
    if (enableTransaction)
    {
        AddLinkCommand* pCmd = new AddLinkCommand(info, this, subGpIdx);
        m_stack->push(pCmd);
        return QModelIndex();
    }
    else
    {
		SubGraphModel* pGraph = subGraph(subGpIdx.row());
        ZASSERT_EXIT(pGraph, QModelIndex());

			QStandardItem* pItem = new QStandardItem;
			pItem->setData(UiHelper::generateUuid(), ROLE_OBJID);
			pItem->setData(info.inputNode, ROLE_INNODE);
			pItem->setData(info.inputSock, ROLE_INSOCK);
			pItem->setData(info.outputNode, ROLE_OUTNODE);
			pItem->setData(info.outputSock, ROLE_OUTSOCK);

			m_linkModel->appendRow(pItem);
			QModelIndex linkIdx = m_linkModel->indexFromItem(pItem);

			const QModelIndex& inIdx = pGraph->index(info.inputNode);
			const QModelIndex& outIdx = pGraph->index(info.outputNode);

			INPUT_SOCKETS inputs = inIdx.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
			OUTPUT_SOCKETS outputs = outIdx.data(ROLE_OUTPUTS).value<OUTPUT_SOCKETS>();
			inputs[info.inputSock].linkIndice.append(QPersistentModelIndex(linkIdx));
			outputs[info.outputSock].linkIndice.append(QPersistentModelIndex(linkIdx));
			pGraph->setData(inIdx, QVariant::fromValue(inputs), ROLE_INPUTS);
			pGraph->setData(outIdx, QVariant::fromValue(outputs), ROLE_OUTPUTS);
            return linkIdx;
    }
}

void GraphsModel::updateLinkInfo(const QPersistentModelIndex& linkIdx, const LINK_UPDATE_INFO& info, bool enableTransaction)
{
    if (enableTransaction)
    {

    }
    else
    {
        m_linkModel->setData(linkIdx, info.newEdge.inputNode, ROLE_INNODE);
        m_linkModel->setData(linkIdx, info.newEdge.inputSock, ROLE_INSOCK);
        m_linkModel->setData(linkIdx, info.newEdge.outputNode, ROLE_OUTNODE);
        m_linkModel->setData(linkIdx, info.newEdge.outputSock, ROLE_OUTSOCK);
    }
}

void GraphsModel::removeSubGraph(const QString& name)
{
	for (int i = 0; i < m_subGraphs.size(); i++)
	{
        if (m_subGraphs[i].pModel->name() == name)
        {
            removeGraph(i);
        }
        else
        {
            m_subGraphs[i].pModel->removeNodeByDescName(name);
        }
	}
}

QVariant GraphsModel::getParamValue(const QString& id, const QString& name, const QModelIndex& subGpIdx)
{
	SubGraphModel* pGraph = subGraph(subGpIdx.row());
    ZASSERT_EXIT(pGraph, QVariant());
    return pGraph->getParamValue(id, name);
}

void GraphsModel::updateParamInfo(const QString& id, PARAM_UPDATE_INFO info, const QModelIndex& subGpIdx, bool enableTransaction)
{
    if (enableTransaction)
    {
        UpdateDataCommand* pCmd = new UpdateDataCommand(id, info, this, subGpIdx);
        m_stack->push(pCmd);
    }
    else
    {
		SubGraphModel* pGraph = subGraph(subGpIdx.row());
        ZASSERT_EXIT(pGraph);
		pGraph->updateParam(id, info.name, info.newValue);

        const QString& nodeName = pGraph->index(id).data(ROLE_OBJNAME).toString();
        if (info.name == "name" && (nodeName == "SubInput" || nodeName == "SubOutput"))
        {
            SOCKET_UPDATE_INFO updateInfo;
            updateInfo.bInput = (nodeName == "SubInput");
            updateInfo.oldInfo.name = info.oldValue.toString();
            updateInfo.newInfo.name = info.newValue.toString();

            updateInfo.updateWay = SOCKET_UPDATE_NAME;
            updateDescInfo(pGraph->name(), updateInfo);
        }
    }
}

void GraphsModel::updateSocket(const QString& nodeid, SOCKET_UPDATE_INFO info, const QModelIndex& subGpIdx, bool enableTransaction)
{
    if (enableTransaction)
    {
        UpdateSocketCommand* pCmd = new UpdateSocketCommand(nodeid, info, this, subGpIdx);
        m_stack->push(pCmd);
    }
    else
    {
		SubGraphModel* pSubg = subGraph(subGpIdx.row());
        ZASSERT_EXIT(pSubg);
		pSubg->updateSocket(nodeid, info);
    }
}

void GraphsModel::updateSocketDefl(const QString& id, PARAM_UPDATE_INFO info, const QModelIndex& subGpIdx, bool enableTransaction)
{
    if (enableTransaction)
    {
        UpdateSockDeflCommand* pCmd = new UpdateSockDeflCommand(id, info, this, subGpIdx);
        m_stack->push(pCmd);
    }
    else
    {
        SubGraphModel *pSubg = subGraph(subGpIdx.row());
        ZASSERT_EXIT(pSubg);
        pSubg->updateSocketDefl(id, info);
    }
}

void GraphsModel::updateNodeStatus(const QString& nodeid, STATUS_UPDATE_INFO info, const QModelIndex& subgIdx, bool enableTransaction)
{
    if (enableTransaction)
    {
        UpdateStateCommand* pCmd = new UpdateStateCommand(nodeid, info, this, subgIdx);
        m_stack->push(pCmd);
    }
    else
    {
		SubGraphModel* pSubg = subGraph(subgIdx.row());
        ZASSERT_EXIT(pSubg);
        pSubg->updateNodeStatus(nodeid, info);
    }
}

void GraphsModel::updateDescInfo(const QString& descName, const SOCKET_UPDATE_INFO& updateInfo)
{
    ZASSERT_EXIT(m_nodesDesc.find(descName) != m_nodesDesc.end());
	NODE_DESC& desc = m_nodesDesc[descName];
	switch (updateInfo.updateWay)
	{
		case SOCKET_INSERT:
		{
            const QString& nameValue = updateInfo.newInfo.name;
            if (updateInfo.bInput)
            {
                // add SubInput
                ZASSERT_EXIT(desc.inputs.find(nameValue) == desc.inputs.end());
                INPUT_SOCKET inputSocket;
                inputSocket.info = updateInfo.newInfo;
                desc.inputs[nameValue] = inputSocket;
            }
            else
            {
                // add SubOutput
                ZASSERT_EXIT(desc.outputs.find(nameValue) == desc.outputs.end());
                OUTPUT_SOCKET outputSocket;
                outputSocket.info = updateInfo.newInfo;
                desc.outputs[nameValue] = outputSocket;
            }
            break;
        }
        case SOCKET_REMOVE:
        {
            const QString& nameValue = updateInfo.newInfo.name;
            if (updateInfo.bInput)
            {
                ZASSERT_EXIT(desc.inputs.find(nameValue) != desc.inputs.end());
                desc.inputs.remove(nameValue);
            }
            else
            {
                ZASSERT_EXIT(desc.outputs.find(nameValue) != desc.outputs.end());
                desc.outputs.remove(nameValue);
            }
            break;
        }
        case SOCKET_UPDATE_NAME:
        {
            const QString& oldName = updateInfo.oldInfo.name;
            const QString& newName = updateInfo.newInfo.name;
            if (updateInfo.bInput)
            {
                ZASSERT_EXIT(desc.inputs.find(oldName) != desc.inputs.end() &&
                    desc.inputs.find(newName) == desc.inputs.end());
                desc.inputs[newName] = desc.inputs[oldName];
                desc.inputs[newName].info.name = newName;
                desc.inputs.remove(oldName);
            }
            else
            {
                ZASSERT_EXIT(desc.outputs.find(oldName) != desc.outputs.end() &&
                    desc.outputs.find(newName) == desc.outputs.end());
                desc.outputs[newName] = desc.outputs[oldName];
                desc.outputs[newName].info.name = newName;
                desc.outputs.remove(oldName);
            }
            break;
        }
        case SOCKET_UPDATE_DEFL:
        {
            break;
        }
        case SOCKET_UPDATE_TYPE:
        {
            break;
        }
    }

    for (int i = 0; i < m_subGraphs.size(); i++)
    {
        SubGraphModel* pModel = m_subGraphs[i].pModel;
        if (pModel->name() != descName)
        {
            QModelIndexList results = pModel->match(index(0, 0), ROLE_OBJNAME, descName, -1, Qt::MatchContains);
            for (auto idx : results)
            {
                const QString& nodeId = idx.data(ROLE_OBJID).toString();
                updateSocket(nodeId, updateInfo, index(i, 0), false);
            }
        }
    }
}

QVariant GraphsModel::getNodeStatus(const QString& id, int role, const QModelIndex& subGpIdx)
{
	SubGraphModel* pGraph = subGraph(subGpIdx.row());
    ZASSERT_EXIT(pGraph, QVariant());
	return pGraph->getNodeStatus(id, role);
}

NODE_DATA GraphsModel::itemData(const QModelIndex& index, const QModelIndex& subGpIdx) const
{
	SubGraphModel* pGraph = subGraph(subGpIdx.row());
    ZASSERT_EXIT(pGraph, NODE_DATA());
    return pGraph->itemData(index);
}

QString GraphsModel::name(const QModelIndex& subGpIdx) const
{
	SubGraphModel* pGraph = subGraph(subGpIdx.row());
    ZASSERT_EXIT(pGraph, "");
    return pGraph->name();
}

void GraphsModel::setName(const QString& name, const QModelIndex& subGpIdx)
{
	SubGraphModel* pGraph = subGraph(subGpIdx.row());
    ZASSERT_EXIT(pGraph);
    pGraph->setName(name);
}

void GraphsModel::replaceSubGraphNode(const QString& oldName, const QString& newName, const QModelIndex& subGpIdx)
{
	SubGraphModel* pGraph = subGraph(subGpIdx.row());
    ZASSERT_EXIT(pGraph);
    pGraph->replaceSubGraphNode(oldName, newName);
}

NODES_DATA GraphsModel::nodes(const QModelIndex& subGpIdx)
{
	SubGraphModel* pGraph = subGraph(subGpIdx.row());
    ZASSERT_EXIT(pGraph, NODES_DATA());
    return pGraph->nodes();
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

void GraphsModel::reload(const QModelIndex& subGpIdx)
{
	SubGraphModel* pGraph = subGraph(subGpIdx.row());
    ZASSERT_EXIT(pGraph);
    //todo
    pGraph->reload();
}

void GraphsModel::onModelInited()
{

}

void GraphsModel::undo()
{
    m_stack->undo();
}

void GraphsModel::redo()
{
    m_stack->redo();
}

QModelIndexList GraphsModel::searchInSubgraph(const QString& objName, const QModelIndex& subgIdx)
{
    SubGraphModel* pModel = subGraph(subgIdx.row());
    return pModel->match(pModel->index(0, 0), ROLE_OBJNAME, objName, -1, Qt::MatchContains);
}

QStandardItemModel* GraphsModel::linkModel() const
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
        SubGraphModel* pSubModel = m_subGraphs[r].pModel;
		if (pSubModel->index(inNode).isValid())
		{
            return index(r, 0);
		}
    }
    return QModelIndex();
}

QGraphicsScene* GraphsModel::scene(const QModelIndex& subgIdx)
{
    ZenoSubGraphScene* pScene = m_subGraphs[subgIdx.row()].pScene;
    if (pScene == nullptr)
    {
        pScene = new ZenoSubGraphScene(this);
        pScene->initModel(subgIdx);
        m_subGraphs[subgIdx.row()].pScene = pScene;
    }
    return pScene;
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


void GraphsModel::onSubInfoChanged(SubGraphModel* pSubModel, const QModelIndex& idx, bool bInput, bool bInsert)
{
    const QString& objId = idx.data(ROLE_OBJID).toString();
    const QString& objName = idx.data(ROLE_OBJNAME).toString();

    PARAMS_INFO params = idx.data(ROLE_PARAMETERS).value<PARAMS_INFO>();
    const QString& nameValue = params["name"].value.toString();

    SOCKET_UPDATE_INFO updateInfo;
    updateInfo.bInput = bInput;
    updateInfo.updateWay = bInsert ? SOCKET_INSERT : SOCKET_REMOVE;

    SOCKET_INFO info;
    info.name = nameValue;
    info.defaultValue = QVariant(); //defl?
    info.type = "";

    updateInfo.oldInfo = updateInfo.newInfo = info;

    const QString& subnetNodeName = pSubModel->name();
    updateDescInfo(subnetNodeName, updateInfo);
}

QList<SEARCH_RESULT> GraphsModel::search(const QString& content, int searchOpts)
{
    QList<SEARCH_RESULT> results;
    if (content.isEmpty())
        return results;

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
    if (searchOpts & SEARCH_NODE)
    {
        for (auto subgInfo : m_subGraphs)
        {
            SubGraphModel* pModel = subgInfo.pModel;
            QModelIndex subgIdx = indexBySubModel(pModel);
            QModelIndexList lst = pModel->match(pModel->index(0, 0), ROLE_OBJNAME, content, -1, Qt::MatchContains);
            for (QModelIndex nodeIdx : lst)
            {
                SEARCH_RESULT result;
                result.targetIdx = nodeIdx;
                result.subgIdx = subgIdx;
                result.type = SEARCH_NODE;
                results.append(result);
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
