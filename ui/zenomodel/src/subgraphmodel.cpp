#include "graphsmodel.h"
#include "subgraphmodel.h"
#include "modelrole.h"
#include "modeldata.h"
#include <zeno/utils/log.h>
#include "uihelper.h"
#include "zassert.h"
#include "parammodel.h"


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

NODE_DATA SubGraphModel::item2NodeData(const _NodeItem& item) const
{
    NODE_DATA data;
    data[ROLE_OBJID] = item.objid;
    data[ROLE_OBJNAME] = item.objCls;
    data[ROLE_OBJPOS] = item.viewpos;
    data[ROLE_COLLASPED] = item.bCollasped;
    data[ROLE_OPTIONS] = item.options;
    data[ROLE_NODETYPE] = item.type;

    INPUT_SOCKETS inputs;
    OUTPUT_SOCKETS outputs;
    PARAMS_INFO params;
    if (item.inputsModel)
        item.inputsModel->getInputSockets(inputs);
    if (item.paramsModel)
        item.paramsModel->getParams(params);
    if (item.outputsModel)
        item.outputsModel->getOutputSockets(outputs);
    data[ROLE_INPUTS] = QVariant::fromValue(inputs);
    data[ROLE_OUTPUTS] = QVariant::fromValue(outputs);
    data[ROLE_PARAMETERS] = QVariant::fromValue(params);

    return data;
}

SubGraphModel::_NodeItem SubGraphModel::nodeData2Item(const NODE_DATA& data, const QModelIndex& nodeIdx)
{
    _NodeItem item;

    item.objid = data[ROLE_OBJID].toString();
    item.objCls = data[ROLE_OBJNAME].toString();
    item.viewpos = data[ROLE_OBJPOS].toPointF();
    item.bCollasped = data[ROLE_COLLASPED].toBool();
    item.options = data[ROLE_OPTIONS].toInt();
    item.type = (NODE_TYPE)data[ROLE_NODETYPE].toInt();

    QModelIndex subgIdx = m_pGraphsModel->indexBySubModel(this);

    INPUT_SOCKETS inputs = data[ROLE_INPUTS].value<INPUT_SOCKETS>();
    PARAMS_INFO params = data[ROLE_PARAMETERS].value<PARAMS_INFO>();
    OUTPUT_SOCKETS outputs = data[ROLE_OUTPUTS].value<OUTPUT_SOCKETS>();

    if (!inputs.isEmpty())
    {
        if (!item.inputsModel)
            item.inputsModel = new IParamModel(PARAM_INPUT, m_pGraphsModel, subgIdx, nodeIdx, this);
        item.inputsModel->setInputSockets(inputs);
    }
    if (!params.isEmpty())
    {
        if (!item.paramsModel)
            item.paramsModel = new IParamModel(PARAM_PARAM, m_pGraphsModel, subgIdx, nodeIdx, this);
        item.paramsModel->setParams(params);
    }
    if (!outputs.isEmpty())
    {
        if (!item.outputsModel)
            item.outputsModel = new IParamModel(PARAM_OUTPUT, m_pGraphsModel, subgIdx, nodeIdx, this);
        item.outputsModel->setOutputSockets(outputs);
    }

    return item;
}

QModelIndex SubGraphModel::index(int row, int column, const QModelIndex& parent) const
{
    if (row < 0 || row >= rowCount())
        return QModelIndex();

    auto itRow = m_row2Key.find(row);
    ZASSERT_EXIT(itRow != m_row2Key.end(), QModelIndex());

    QString nodeid = itRow.value();
    auto itItem = m_nodes.find(nodeid);
    ZASSERT_EXIT(itItem != m_nodes.end(), QModelIndex());

    uint32_t interlId = (m_str2numId.find(nodeid) != m_str2numId.end()) ? m_str2numId[nodeid] : 0;
    return createIndex(row, 0, interlId);
}

QModelIndex SubGraphModel::index(QString id, const QModelIndex& parent) const
{
    auto it = m_nodes.find(id);
    if (it == m_nodes.end())
        return QModelIndex();

    int row = m_key2Row[id];
    uint32_t interlId = (m_str2numId.find(id) != m_str2numId.end()) ? m_str2numId[id] : 0;
    return createIndex(row, 0, interlId);
}

QModelIndex SubGraphModel::index(int id) const
{
    if (m_num2strId.find(id) == m_num2strId.end())
        return QModelIndex();
    return index(m_num2strId[id]);
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
    _NodeItem item;
    if (!itemFromIndex(index, item))
        return false;

    QString currNode = index.data(ROLE_OBJID).toString();
    const QModelIndex& subgIdx = m_pGraphsModel->indexBySubModel(this);

    if (item.inputsModel)
    {
        item.inputsModel->clear();
        delete item.inputsModel;
        item.inputsModel = nullptr;
    }
    if (item.paramsModel)
    {
        item.paramsModel->clear();
        delete item.paramsModel;
        item.paramsModel = nullptr;
    }
    if (item.outputsModel)
    {
        item.outputsModel->clear();
        delete item.outputsModel;
        item.outputsModel = nullptr;
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

    uint32_t numId = m_str2numId[id];
    m_num2strId.remove(numId);
    m_str2numId.remove(id);

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

IParamModel* SubGraphModel::paramModel(const QModelIndex& index, PARAM_CLASS cls) const
{
    _NodeItem item;
    if (!itemFromIndex(index, item))
        return nullptr;

    switch (cls)
    {
    case PARAM_INPUT: return item.inputsModel;
    case PARAM_PARAM: return item.paramsModel;
    case PARAM_OUTPUT: return item.outputsModel;
    default:
        return nullptr;
    }
}

QVariant SubGraphModel::data(const QModelIndex& index, int role) const
{
    _NodeItem item;
    if (!itemFromIndex(index, item))
        return QVariant();

    switch (role)
    {
        case ROLE_OBJID:    return item.objid;
        case ROLE_OBJNAME:  return item.objCls;
        case ROLE_NODETYPE: return item.type;
        case ROLE_INPUTS:
        {
            //legacy interface.
            if (!item.inputsModel)
                return QVariant();

            INPUT_SOCKETS inputs;
            item.inputsModel->getInputSockets(inputs);
            return QVariant::fromValue(inputs);
        }
        case ROLE_OUTPUTS:
        {
            if (!item.outputsModel)
                return QVariant();

            OUTPUT_SOCKETS outputs;
            item.outputsModel->getOutputSockets(outputs);
            return QVariant::fromValue(outputs);
        }
        case ROLE_PARAMETERS:
        {
            if (!item.paramsModel)
                return QVariant();

            PARAMS_INFO params;
            item.paramsModel->getParams(params);
            return QVariant::fromValue(params);
        }
        case ROLE_COLLASPED:
        {
            return item.bCollasped;
        }
        case ROLE_OPTIONS:
        {
            return item.options;
        }
        case ROLE_OBJPOS:
        {
            return item.viewpos;
        }
        default:
            return QVariant();
    }
}

bool SubGraphModel::setData(const QModelIndex& index, const QVariant& value, int role)
{
    QString id = m_row2Key[index.row()];
    if (m_nodes.find(id) != m_nodes.end())
    {
        _NodeItem& item = m_nodes[id];

        switch (role)
        {
            case ROLE_INPUTS:
            {
                INPUT_SOCKETS inputs = value.value<INPUT_SOCKETS>();
                if (inputs.empty())
                    return false;

                if (!item.inputsModel)
                    item.inputsModel = new IParamModel(PARAM_INPUT, m_pGraphsModel, m_pGraphsModel->indexBySubModel(this), index, this);

                for (QString name : inputs.keys())
                {
                    const INPUT_SOCKET& inSocket = inputs[name];
                    QModelIndex paramIdx = item.inputsModel->index(name);
                    if (!paramIdx.isValid())
                    {
                        item.inputsModel->appendRow(name, inSocket.info.type, inSocket.info.defaultValue, inSocket.info.control, inSocket.linkIndice);
                    }
                    else
                    {
                        item.inputsModel->setItem(paramIdx, inSocket.info.type, inSocket.info.defaultValue, inSocket.info.control, inSocket.linkIndice);
                    }
                }
                break;
            }
            case ROLE_OUTPUTS:
            {
                OUTPUT_SOCKETS outputs = value.value<OUTPUT_SOCKETS>();
                if (outputs.empty())
                    return false;

                if (!item.outputsModel)
                    item.outputsModel = new IParamModel(PARAM_OUTPUT, m_pGraphsModel, m_pGraphsModel->indexBySubModel(this), index, this);

                for (QString name : outputs.keys())
                {
                    const OUTPUT_SOCKET& outSocket = outputs[name];
                    QModelIndex idx = item.outputsModel->index(name);
                    if (!idx.isValid())
                    {
                        item.outputsModel->appendRow(name, outSocket.info.type, outSocket.info.defaultValue, outSocket.info.control, outSocket.linkIndice);
                    }
                    else
                    {
                        item.outputsModel->setItem(idx, outSocket.info.type, outSocket.info.defaultValue, outSocket.info.control, outSocket.linkIndice);
                    }
                }
                break;
            }
            case ROLE_PARAMETERS:
            {
                PARAMS_INFO params = value.value<PARAMS_INFO>();
                if (params.empty())
                    return false;

                if (!item.paramsModel)
                    item.paramsModel = new IParamModel(PARAM_PARAM, m_pGraphsModel, m_pGraphsModel->indexBySubModel(this), index, this);

                for (QString name : params.keys())
                {
                    const PARAM_INFO& param = params[name];
                    QModelIndex idx = item.paramsModel->index(name);
                    if (!idx.isValid())
                    {
                        item.paramsModel->appendRow(name, param.typeDesc, param.value, param.control);
                    }
                    else
                    {
                        item.paramsModel->setItem(idx, param.typeDesc, param.value, param.control);
                    }
                }
                break;
            }
            case ROLE_COLLASPED:
            {
                item.bCollasped = value.toBool();
                break;
            }
            case ROLE_OPTIONS:
            {
                item.options = value.toInt();
                break;
            }
            case ROLE_OBJPOS:
            {
                item.viewpos = value.toPointF();
                break;
            }
        }

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

void SubGraphModel::updateParam(const QString& nodeid, const QString& paramName, const QVariant& var, QString* newType)
{
    auto it = m_nodes.find(nodeid);
    if (it == m_nodes.end())
        return;

    const QModelIndex& idx = index(nodeid);

    IParamModel* pModel = m_nodes[nodeid].paramsModel;
    ZASSERT_EXIT(pModel);

    QModelIndex paramIdx = pModel->index(paramName);
    if (!paramIdx.isValid())
        return;

    const QString& nodeCls = idx.data(ROLE_OBJNAME).toString();
    //correct the control type and desc type according to the type of real value.
    if (paramName == "defl" && (nodeCls == "SubInput" || nodeCls == "SubOutput") && newType)
    {
        setData(paramIdx, *newType, ROLE_PARAM_TYPE);
        setData(paramIdx, UiHelper::getControlType(*newType), ROLE_PARAM_CTRL);
    }
    setData(paramIdx, var, ROLE_PARAM_VALUE);
}

void SubGraphModel::updateSocket(const QString& nodeid, const SOCKET_UPDATE_INFO& info)
{
    QModelIndex idx = index(nodeid);
    ZASSERT_EXIT(idx.isValid());

    _NodeItem item;
    if (!itemFromIndex(idx, item))
    {
        ZASSERT_EXIT(false);
        return;
    }

    IParamModel* pModel = info.bInput ? item.inputsModel : item.outputsModel;
    ZASSERT_EXIT(pModel);

    //INPUT_SOCKETS inputs = data(idx, ROLE_INPUTS).value<INPUT_SOCKETS>();
    const QString& nodeName = idx.data(ROLE_OBJNAME).toString();
    const QString& oldName = info.oldInfo.name;
    const QString& newName = info.newInfo.name;
    switch (info.updateWay)
    {
        case SOCKET_INSERT:
        {
            if (nodeName == "MakeDict" || nodeName == "ExtractDict" || nodeName == "MakeList")
            {
                //dynamic socket in dict grows by bottom direction.
                pModel->appendRow(newName, info.newInfo.type, info.newInfo.defaultValue, info.newInfo.control);
            }
            else
            {
                pModel->insertRow(0, newName, info.newInfo.type, info.newInfo.defaultValue, info.newInfo.control);
            }
            break;
        }
        case SOCKET_REMOVE:
        {
            QModelIndex paramIdx = pModel->index(oldName);
            ZASSERT_EXIT(paramIdx.isValid());
            bool ret = pModel->removeRow(paramIdx.row());
            ZASSERT_EXIT(ret);
            break;
        }
        case SOCKET_UPDATE_NAME:
        {
            QModelIndex paramIdx = pModel->index(oldName);
            ZASSERT_EXIT(paramIdx.isValid());
            bool ret = pModel->setData(paramIdx, newName, ROLE_PARAM_NAME);
            ZASSERT_EXIT(ret);
            break;
        }
        case SOCKET_UPDATE_TYPE:
        {
            QModelIndex paramIdx = pModel->index(oldName);
            ZASSERT_EXIT(paramIdx.isValid());
            //todo: unify the operation together.
            pModel->setData(paramIdx, info.newInfo.type, ROLE_PARAM_TYPE);
            pModel->setData(paramIdx, info.newInfo.control, ROLE_PARAM_CTRL);
            pModel->setData(paramIdx, info.newInfo.defaultValue, ROLE_PARAM_VALUE);
            break;
        }
        case SOCKET_UPDATE_DEFL:
        {
            QModelIndex paramIdx = pModel->index(oldName);
            ZASSERT_EXIT(paramIdx.isValid());
            pModel->setData(paramIdx, info.newInfo.defaultValue, ROLE_PARAM_VALUE);
            break;
        }
    }
}

void SubGraphModel::updateSocketDefl(const QString& nodeid, const PARAM_UPDATE_INFO& info)
{
	QModelIndex idx = index(nodeid);
    ZASSERT_EXIT(idx.isValid());

    _NodeItem item;
    if (!itemFromIndex(idx, item))
        return;

    if (IParamModel* pModel = item.inputsModel)
    {
        QModelIndex paramIdx = pModel->index(info.name);
        bool ret = pModel->setData(paramIdx, info.newValue, ROLE_PARAM_VALUE);
        if (ret) {}
    }
}

QVariant SubGraphModel::getParamValue(const QString& nodeid, const QString& paramName)
{
    QVariant var;
    if (IParamModel* pModel = m_nodes[nodeid].paramsModel)
    {
        QModelIndex idx = pModel->index(paramName);
        if (idx.isValid())
            var = idx.data(ROLE_PARAM_VALUE);
    }
    return var;
}

void SubGraphModel::updateNodeStatus(const QString& nodeid, STATUS_UPDATE_INFO info)
{
    auto it = m_nodes.find(nodeid);
    if (it == m_nodes.end())
        return;

    _NodeItem& item = m_nodes[nodeid];
    switch (info.role)
    {
    case ROLE_COLLASPED: item.bCollasped = info.newValue.toBool(); break;
    case ROLE_OPTIONS: item.options = info.newValue.toInt(); break;
    case ROLE_OBJPOS: item.viewpos = info.newValue.toPointF(); break;
    default:
        ZASSERT_EXIT(false);
        break;
    }

    const QModelIndex& idx = index(nodeid);
    emit dataChanged(idx, idx, QVector<int>{info.role});
}

bool SubGraphModel::hasChildren(const QModelIndex& parent) const
{
    return false;
}

NODE_DATA SubGraphModel::nodeData(const QModelIndex &index) const
{
    _NodeItem item;
    if (!itemFromIndex(index, item))
        return NODE_DATA();
    return item2NodeData(item);
}

QModelIndexList SubGraphModel::match(const QModelIndex& start, int role, const QVariant& value, int hits, Qt::MatchFlags flags) const
{
    return _base::match(start, role, value, hits, flags);
}

bool SubGraphModel::itemFromIndex(const QModelIndex &index, _NodeItem& retNode) const
{
    if (!index.isValid())
        return false;

    if (m_row2Key.find(index.row()) == m_row2Key.end())
        return false;

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
        m_row2Key[nRows] = id;
        m_key2Row[id] = nRows;
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
        m_row2Key[row] = id;
        m_key2Row[id] = row;
    }
    else
    {
        Q_ASSERT(false);
        return false;
    }

    _NodeItem& item = m_nodes[id];
    QModelIndex nodeIdx = index(row, 0, QModelIndex());
    QModelIndex subgIdx = m_pGraphsModel->indexBySubModel(this);

    item = nodeData2Item(nodeData, nodeIdx);

    QUuid uuid = QUuid::createUuid();
    uint32_t ident = uuid.data1;
    m_num2strId[ident] = id;
    m_str2numId[id] = ident;
    m_pGraphsModel->markDirty();
    return true;
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
