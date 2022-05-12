#include "graphsmodel.h"
#include "subgraphmodel.h"
#include <zenoui/model/modelrole.h>
#include <zenoui/model/modeldata.h>
#include <zeno/utils/log.h>
#include <zenoui/util/uihelper.h>
#include "util/apphelper.h"
#include "util/log.h"


SubGraphModel::SubGraphModel(GraphsModel* pGraphsModel, QObject *parent)
    : QAbstractItemModel(pGraphsModel)
    , m_pGraphsModel(pGraphsModel)
    , m_stack(new QUndoStack(this))
{
	connect(this, &QAbstractItemModel::dataChanged, m_pGraphsModel, &GraphsModel::on_subg_dataChanged);
	connect(this, &QAbstractItemModel::rowsAboutToBeInserted, m_pGraphsModel, &GraphsModel::on_subg_rowsAboutToBeInserted);
	connect(this, &QAbstractItemModel::rowsInserted, m_pGraphsModel, &GraphsModel::on_subg_rowsInserted);
	connect(this, &QAbstractItemModel::rowsAboutToBeRemoved, m_pGraphsModel, &GraphsModel::on_subg_rowsAboutToBeRemoved);
	connect(this, &QAbstractItemModel::rowsRemoved, m_pGraphsModel, &GraphsModel::on_subg_rowsRemoved);
}

SubGraphModel::~SubGraphModel()
{
    clear();
}

SubGraphModel::SubGraphModel(const SubGraphModel &rhs)
    : m_pGraphsModel(nullptr)
    , m_stack(new QUndoStack(this))
    , m_key2Row(rhs.m_key2Row)
    , m_row2Key(rhs.m_row2Key)
    , m_rect(rhs.m_rect)
    , m_name(rhs.m_name)
    , m_nodes(rhs.m_nodes)
{
}

void SubGraphModel::onModelInited()
{
}

void SubGraphModel::setInputSocket(const QString& inNode, const QString& inSock, const QString& outId, const QString& outSock, const QVariant& defaultValue)
{
	QModelIndex idx = index(inNode);
    ZASSERT_EXIT(idx.isValid());
	INPUT_SOCKETS inputs = data(idx, ROLE_INPUTS).value<INPUT_SOCKETS>();
	if (inputs.find(inSock) != inputs.end())
	{
		if (!defaultValue.isNull())
			inputs[inSock].info.defaultValue = defaultValue;	//default value?
		if (!outId.isEmpty() && !outSock.isEmpty())
		{
            SOCKET_INFO info(outId, outSock);

            //because of initialization, we had to add index to input socket first,
            //later the index will be appended into output socket.
            QStandardItemModel* pModel = m_pGraphsModel->linkModel();   //it's not a good habit to expose linkModel

            QStandardItem* pItem = new QStandardItem;
            pItem->setData(UiHelper::generateUuid(), ROLE_OBJID);
            pItem->setData(inNode, ROLE_INNODE);
            pItem->setData(inSock, ROLE_INSOCK);
            pItem->setData(outId, ROLE_OUTNODE);
            pItem->setData(outSock, ROLE_OUTSOCK);
            pModel->appendRow(pItem);

            QModelIndex linkIdx = pModel->indexFromItem(pItem);
            inputs[inSock].linkIndice.push_back(QPersistentModelIndex(linkIdx));
			setData(idx, QVariant::fromValue(inputs), ROLE_INPUTS);
		}
	}
}

void SubGraphModel::collaspe()
{
	for (int i = 0; i < rowCount(); i++)
	{
		setData(index(i, 0), true, ROLE_COLLASPED);
	}
}

void SubGraphModel::expand()
{
    for (int i = 0; i < rowCount(); i++)
    {
        setData(index(i, 0), false, ROLE_COLLASPED);
    }
}

NODES_DATA SubGraphModel::nodes()
{
    NODES_DATA datas;
    for (auto iter = m_nodes.keyValueBegin(); iter != m_nodes.keyValueEnd(); iter++)
    {
        datas[(*iter).first] = (*iter).second;//cihou wendous, he doesn't support ->
    }
    return datas;
}

void SubGraphModel::clear()
{
    m_nodes.clear();
    m_key2Row.clear();
    m_row2Key.clear();
    const QModelIndex& subgIdx = m_pGraphsModel->indexBySubModel(this);
    emit m_pGraphsModel->clearLayout(subgIdx);
}

void SubGraphModel::reload()
{
    const QModelIndex& subgIdx = m_pGraphsModel->indexBySubModel(this);
    emit m_pGraphsModel->reloaded(subgIdx);
}

QModelIndex SubGraphModel::index(int row, int column, const QModelIndex& parent) const
{
    if (row < 0 || row >= rowCount())
        return QModelIndex();

    auto itRow = m_row2Key.find(row);
    ZASSERT_EXIT(itRow != m_row2Key.end(), QModelIndex());

    auto itItem = m_nodes.find(itRow.value());
    ZASSERT_EXIT(itItem != m_nodes.end(), QModelIndex());

    return createIndex(row, 0, nullptr);
}

QModelIndex SubGraphModel::index(QString id, const QModelIndex& parent) const
{
    auto it = m_nodes.find(id);
    if (it == m_nodes.end())
        return QModelIndex();

    int row = m_key2Row[id];
    return createIndex(row, 0, nullptr);
}

void SubGraphModel::appendItem(const NODE_DATA& nodeData, bool enableTransaction)
{
    //called on both right-click and load-zsg, both enabletrans=false
    //zeno::log_warn("both has Inputs {}", nodeData.find(ROLE_PARAMETERS) != nodeData.end());
    int nRow = m_nodes.size();
    insertRow(nRow, nodeData);
}

void SubGraphModel::removeNode(int row, bool enableTransaction)
{
	removeNode(m_row2Key[row], enableTransaction);
}

void SubGraphModel::removeNode(const QString& nodeid, bool enableTransaction)
{
    ZASSERT_EXIT(m_key2Row.find(nodeid) != m_key2Row.end());
    int row = m_key2Row[nodeid];
    removeRows(row, 0);
}

bool SubGraphModel::removeRows(int row, int count, const QModelIndex& parent)
{
	beginRemoveRows(parent, row, row);
	_removeRow(index(row, 0));
	endRemoveRows();
	return true;
}

bool SubGraphModel::_removeRow(const QModelIndex& index)
{
    //remove node by id and update params from other node.
    NODE_DATA nodeData;
    if (!itemFromIndex(index, nodeData))
        return false;

    QString currNode = index.data(ROLE_OBJID).toString();
    const QModelIndex& subgIdx = m_pGraphsModel->indexBySubModel(this);

    const INPUT_SOCKETS& inputs = nodeData[ROLE_INPUTS].value<INPUT_SOCKETS>();
    for (QString inSock : inputs.keys())
    {
        INPUT_SOCKET inputSocket = inputs[inSock];
        m_pGraphsModel->removeLinks(inputSocket.linkIndice, subgIdx, true);
    }

    // in this loop, output refers to current node's output, input refers to what output points to.
    const OUTPUT_SOCKETS& outputs = index.data(ROLE_OUTPUTS).value<OUTPUT_SOCKETS>();
    for (QString outSock : outputs.keys())
    {
        OUTPUT_SOCKET outputSocket = outputs[outSock];
        m_pGraphsModel->removeLinks(outputSocket.linkIndice, subgIdx, true);
    }

    int row = index.row();
    QString id = m_row2Key[row];
    ZASSERT_EXIT(!id.isEmpty(), false);
    for (int r = row + 1; r < rowCount(); r++)
    {
        const QString &key = m_row2Key[r];
        m_row2Key[r - 1] = key;
        m_key2Row[key] = r - 1;
    }

    m_row2Key.remove(rowCount() - 1);
    m_key2Row.remove(id);
    m_nodes.remove(id);
    m_pGraphsModel->markDirty();
    return true;
}

void SubGraphModel::removeNodeByDescName(const QString& descName)
{
    QStringList nodes;
    for (int r = 0; r < rowCount(); r++)
    {
        const QModelIndex& idx = index(r, 0);
        const QString& objName = data(idx, ROLE_OBJNAME).toString();
        if (objName == descName)
        {
            nodes.push_back(data(idx, ROLE_OBJID).toString());
        }
    }
    for (const QString& nodeid : nodes)
    {
        removeNode(nodeid);
    }
}

QModelIndex SubGraphModel::parent(const QModelIndex& child) const
{
    return QModelIndex();
}

int SubGraphModel::rowCount(const QModelIndex& parent) const
{
    return m_nodes.size();
}

int SubGraphModel::columnCount(const QModelIndex& parent) const
{
	return 1;
}

QVariant SubGraphModel::data(const QModelIndex& index, int role) const
{
    NODE_DATA nodeData;
    if (!itemFromIndex(index, nodeData))
        return QVariant();

    if (role == ROLE_INPUTS)
    {
        //get input sockets from core.
    }

    return nodeData[role];
}

bool SubGraphModel::setData(const QModelIndex& index, const QVariant& value, int role)
{
    QString id = m_row2Key[index.row()];
    if (m_nodes.find(id) != m_nodes.end())
    {
        m_nodes[id][role] = value;
        emit dataChanged(index, index, QVector<int>{role});
        m_pGraphsModel->markDirty();
        return true;
    }
    else
    {
        return false;
    }
}

SubGraphModel* SubGraphModel::clone(GraphsModel* parent)
{
    SubGraphModel *pClone = new SubGraphModel(*this);
    pClone->m_pGraphsModel = parent;
    return pClone;
}

void SubGraphModel::updateParam(const QString& nodeid, const QString& paramName, const QVariant& var, bool enableTransaction)
{
    auto it = m_nodes.find(nodeid);
    if (it == m_nodes.end())
        return;

    PARAMS_INFO params = m_nodes[nodeid][ROLE_PARAMETERS].value<PARAMS_INFO>();
    const QVariant oldValue = params[paramName].value;
    QVariant newValue = var;
    params[paramName].value = newValue;

    const QModelIndex &idx = index(nodeid);
    const QString &nodeName = idx.data(ROLE_OBJNAME).toString();

    const QModelIndex& subgIdx = m_pGraphsModel->indexBySubModel(this);
    //for SubInput and SubOutput£¬the "name" value shoudle be checked.
    AppHelper::correctSubIOName(m_pGraphsModel, subgIdx, nodeName, params);
    setData(idx, QVariant::fromValue(params), ROLE_PARAMETERS);
    //emit dataChanged signal, notify ui view to sync.
    setData(idx, QVariant::fromValue(params[paramName]), ROLE_MODIFY_PARAM);  
}

void SubGraphModel::updateSocket(const QString& nodeid, const SOCKET_UPDATE_INFO& info)
{
    if (info.bInput)
    {
        QModelIndex idx = index(nodeid);
        INPUT_SOCKETS inputs = data(idx, ROLE_INPUTS).value<INPUT_SOCKETS>();
        const QString& oldName = info.oldInfo.name;
        const QString& newName = info.newInfo.name;
        switch (info.updateWay)
        {
            case SOCKET_INSERT:
            {
                INPUT_SOCKET newSock;
                newSock.info = info.newInfo;
                inputs[newName] = newSock;
                setData(idx, QVariant::fromValue(inputs), ROLE_INPUTS);
                setData(idx, QVariant::fromValue(info), ROLE_MODIFY_SOCKET);
                break;
            }
            case SOCKET_REMOVE:
            {
                ZASSERT_EXIT(inputs.find(oldName) != inputs.end());
                INPUT_SOCKET inputSock = inputs[oldName];
				//remove link connected to oldName.
				for (QPersistentModelIndex linkIdx : inputSock.linkIndice)
				{
					const QModelIndex& subgIdx = m_pGraphsModel->indexBySubModel(this);
					m_pGraphsModel->removeLink(linkIdx, subgIdx, false);
				}
                inputs.remove(oldName);
				setData(idx, QVariant::fromValue(inputs), ROLE_INPUTS);
				setData(idx, QVariant::fromValue(info), ROLE_MODIFY_SOCKET);
                break;
            }
            case SOCKET_UPDATE_NAME:
            {
                ZASSERT_EXIT(inputs.find(oldName) != inputs.end());

			    INPUT_SOCKET inputSock = inputs[oldName];
			    inputSock.info.name = newName;

			    inputs.remove(oldName);
			    inputs[newName] = inputSock;
			    setData(idx, QVariant::fromValue(inputs), ROLE_INPUTS);
			    setData(idx, QVariant::fromValue(info), ROLE_MODIFY_SOCKET);    //will sync to scene/node.
                //todo: The sync by calling setData is somehow dangerous, we need to fetch scene and update node directly.

			    for (QPersistentModelIndex linkIdx : inputSock.linkIndice)
			    {
				    //modify link info.
				    QString outNode = linkIdx.data(ROLE_OUTNODE).toString();
				    QString outSock = linkIdx.data(ROLE_OUTSOCK).toString();
				    QString inNode = linkIdx.data(ROLE_INNODE).toString();
				    QString inSock = linkIdx.data(ROLE_INSOCK).toString();

				    ZASSERT_EXIT(inSock == oldName);
				    LINK_UPDATE_INFO updateInfo;
				    updateInfo.oldEdge = EdgeInfo(outNode, inNode, outSock, inSock);
				    updateInfo.newEdge = EdgeInfo(outNode, inNode, outSock, newName);

				    m_pGraphsModel->updateLinkInfo(linkIdx, updateInfo, false);
			    }
                break;
            }
            case SOCKET_UPDATE_TYPE:
            case SOCKET_UPDATE_DEFL:
            {
			    INPUT_SOCKET& inputSock = inputs[oldName];
			    inputSock.info = info.newInfo;
			    setData(idx, QVariant::fromValue(inputs), ROLE_INPUTS);
			    setData(idx, QVariant::fromValue(info), ROLE_MODIFY_SOCKET);
                break;
            }
        }
    }
    else
    {
        QModelIndex idx = index(nodeid);
        OUTPUT_SOCKETS outputs = data(idx, ROLE_OUTPUTS).value<OUTPUT_SOCKETS>();
        const QString& oldName = info.oldInfo.name;
        const QString& newName = info.newInfo.name;

        switch (info.updateWay)
        {
            case SOCKET_INSERT:
            {
				OUTPUT_SOCKET newSock;
				newSock.info = info.newInfo;
				outputs[newName] = newSock;
				setData(idx, QVariant::fromValue(outputs), ROLE_OUTPUTS);
				setData(idx, QVariant::fromValue(info), ROLE_MODIFY_SOCKET);
				break;
            }
            case SOCKET_REMOVE:
            {
                ZASSERT_EXIT(outputs.find(oldName) != outputs.end());
				OUTPUT_SOCKET outputSock = outputs[oldName];
				//remove link connected to oldName.
				for (QPersistentModelIndex linkIdx : outputSock.linkIndice)
				{
					const QModelIndex& subgIdx = m_pGraphsModel->indexBySubModel(this);
					m_pGraphsModel->removeLink(linkIdx, subgIdx, false);
				}
                outputs.remove(oldName);
				setData(idx, QVariant::fromValue(outputs), ROLE_OUTPUTS);
				setData(idx, QVariant::fromValue(info), ROLE_MODIFY_SOCKET);
				break;
            }
            case SOCKET_UPDATE_NAME:
            {
                ZASSERT_EXIT(outputs.find(oldName) != outputs.end());
				OUTPUT_SOCKET outputSock = outputs[oldName];
				outputSock.info.name = newName;

				outputs.remove(oldName);
				outputs[newName] = outputSock;
				setData(idx, QVariant::fromValue(outputs), ROLE_OUTPUTS);
				setData(idx, QVariant::fromValue(info), ROLE_MODIFY_SOCKET);

				for (QPersistentModelIndex linkIdx : outputSock.linkIndice)
				{
					//modify link info.
					QString outNode = linkIdx.data(ROLE_OUTNODE).toString();
					QString outSock = linkIdx.data(ROLE_OUTSOCK).toString();
					QString inNode = linkIdx.data(ROLE_INNODE).toString();
					QString inSock = linkIdx.data(ROLE_INSOCK).toString();

					ZASSERT_EXIT(outSock == oldName);
					LINK_UPDATE_INFO updateInfo;
					updateInfo.oldEdge = EdgeInfo(outNode, inNode, outSock, inSock);
					updateInfo.newEdge = EdgeInfo(outNode, inNode, newName, inSock);

					m_pGraphsModel->updateLinkInfo(linkIdx, updateInfo, false);
				}
                break;
            }
            case SOCKET_UPDATE_TYPE:
            {

            }
            case SOCKET_UPDATE_DEFL:
            {
				OUTPUT_SOCKET& outputSock = outputs[oldName];
				outputSock.info = info.newInfo;
				setData(idx, QVariant::fromValue(outputs), ROLE_OUTPUTS);
				setData(idx, QVariant::fromValue(info), ROLE_MODIFY_SOCKET);
                break;
            }
        }
    }
}

void SubGraphModel::updateSocketDefl(const QString& nodeid, const PARAM_UPDATE_INFO& info)
{
	QModelIndex idx = index(nodeid);
    ZASSERT_EXIT(idx.isValid());
	INPUT_SOCKETS inputs = data(idx, ROLE_INPUTS).value<INPUT_SOCKETS>();
    ZASSERT_EXIT(inputs.find(info.name) != inputs.end());
    inputs[info.name].info.defaultValue = info.newValue;
    setData(idx, QVariant::fromValue(inputs), ROLE_INPUTS);
    setData(idx, QVariant::fromValue(info), ROLE_MODIFY_SOCKET_DEFL);
}

QVariant SubGraphModel::getParamValue(const QString& nodeid, const QString& paramName)
{
    QVariant var;
    const PARAMS_INFO info = m_nodes[nodeid][ROLE_PARAMETERS].value<PARAMS_INFO>();
    return info[paramName].value;
}

QVariant SubGraphModel::getNodeStatus(const QString& nodeid, int role)
{
    ZASSERT_EXIT(m_nodes.find(nodeid) != m_nodes.end(), QVariant());
    return m_nodes[nodeid][role];
}

void SubGraphModel::updateNodeStatus(const QString& nodeid, STATUS_UPDATE_INFO info)
{
    auto it = m_nodes.find(nodeid);
    if (it == m_nodes.end())
        return;

    m_nodes[nodeid][info.role] = info.newValue;
    const QModelIndex& idx = index(nodeid);
    emit dataChanged(idx, idx, QVector<int>{info.role});
}

bool SubGraphModel::hasChildren(const QModelIndex& parent) const
{
    return false;
}

QVariant SubGraphModel::headerData(int section, Qt::Orientation orientation, int role) const
{
    return _base::headerData(section, orientation, role);
}

bool SubGraphModel::setHeaderData(int section, Qt::Orientation orientation, const QVariant &value, int role)
{
    return _base::setHeaderData(section, orientation, value, role);
}

NODE_DATA SubGraphModel::itemData(const QModelIndex &index) const
{
    NODE_DATA nodeData;
    if (!itemFromIndex(index, nodeData))
        return NODE_DATA();
    return nodeData;
}

QModelIndexList SubGraphModel::match(const QModelIndex& start, int role, const QVariant& value, int hits, Qt::MatchFlags flags) const
{
    return _base::match(start, role, value, hits, flags);
}

QHash<int, QByteArray> SubGraphModel::roleNames() const
{
    return _base::roleNames();
}

bool SubGraphModel::itemFromIndex(const QModelIndex &index, NODE_DATA &retNode) const
{
    QString id = m_row2Key[index.row()];
    if (m_nodes.find(id) != m_nodes.end())
    {
        retNode = m_nodes[id];
        return true;
    }
    else
    {
        return false;
    }
}

bool SubGraphModel::_insertRow(int row, const NODE_DATA& nodeData, const QModelIndex &parent)
{
    //pure insert logic, without transaction and notify stuffs.
    const QString& id = nodeData[ROLE_OBJID].toString();
    const QString& name = nodeData[ROLE_OBJNAME].toString();

    ZASSERT_EXIT(!id.isEmpty() && !name.isEmpty() && m_nodes.find(id) == m_nodes.end(), false);
    int nRows = m_nodes.size();
    if (row == nRows)
    {
        //append
        m_nodes[id] = nodeData;
        m_row2Key[nRows] = id;
        m_key2Row[id] = nRows;
        m_pGraphsModel->markDirty();
        return true;
    }
    else if (row < nRows)
    {
        auto itRow = m_row2Key.find(row);
        ZASSERT_EXIT(itRow != m_row2Key.end(), false);
        int nRows = rowCount();
        for (int r = nRows; r > row; r--)
        {
            const QString &key = m_row2Key[r - 1];
            m_row2Key[r] = key;
            m_key2Row[key] = r;
        }
        m_nodes[id] = nodeData;
        m_row2Key[row] = id;
        m_key2Row[id] = row;
        m_pGraphsModel->markDirty();
        return true;
    }
    else
    {
        Q_ASSERT(false);
        return false;
    }
}

bool SubGraphModel::insertRow(int row, const NODE_DATA &nodeData, const QModelIndex &parent)
{
    beginInsertRows(QModelIndex(), row, row);
    bool ret = _insertRow(row, nodeData);
    endInsertRows();
    return ret;
}

void SubGraphModel::onDoubleClicked(const QString& nodename)
{
    m_pGraphsModel->switchSubGraph(nodename);
}

void SubGraphModel::setName(const QString& name)
{
    m_name = name;
}

void SubGraphModel::setViewRect(const QRectF& rc)
{
    m_rect = rc;
}

QString SubGraphModel::name() const
{
    return m_name;
}

void SubGraphModel::replaceSubGraphNode(const QString& oldName, const QString& newName)
{
    for (int i = 0; i < rowCount(); i++)
    {
        const QModelIndex& idx = index(i, 0);
        if (idx.data(ROLE_OBJNAME).toString() == oldName)
        {
            setData(idx, newName, ROLE_OBJNAME);
        }
    }
}
