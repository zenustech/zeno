#include "graphsmodel.h"
#include "subgraphmodel.h"
#include "modelrole.h"
#include "modeldata.h"
#include <zeno/utils/log.h>
#include "uihelper.h"
#include "zassert.h"
#include "parammodel.h"
#include "variantptr.h"



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

    item.nodeParams->getInputSockets(inputs);
    item.nodeParams->getParams(params);
    item.nodeParams->getOutputSockets(outputs);

    data[ROLE_INPUTS] = QVariant::fromValue(inputs);
    data[ROLE_OUTPUTS] = QVariant::fromValue(outputs);
    data[ROLE_PARAMETERS] = QVariant::fromValue(params);
    data[ROLE_PANEL_PARAMS] = QVariantPtr<ViewParamModel>::asVariant(item.panelParams);
    data[ROLE_PARAMS_NO_DESC] = QVariant::fromValue(item.paramNotDesc);

    return data;
}

void SubGraphModel::importNodeItem(const NODE_DATA& data, const QModelIndex& nodeIdx, _NodeItem& ret)
{
    ret.objid = data[ROLE_OBJID].toString();
    ret.objCls = data[ROLE_OBJNAME].toString();
    ret.viewpos = data[ROLE_OBJPOS].toPointF();
    ret.bCollasped = data[ROLE_COLLASPED].toBool();
    ret.options = data[ROLE_OPTIONS].toInt();
    ret.type = (NODE_TYPE)data[ROLE_NODETYPE].toInt();
    ret.paramNotDesc = data[ROLE_PARAMS_NO_DESC].value<PARAMS_INFO>();

    QModelIndex subgIdx = m_pGraphsModel->indexBySubModel(this);

    ret.nodeParams = new NodeParamModel(subgIdx, nodeIdx, m_pGraphsModel, false, this);

    INPUT_SOCKETS inputs = data[ROLE_INPUTS].value<INPUT_SOCKETS>();
    PARAMS_INFO params = data[ROLE_PARAMETERS].value<PARAMS_INFO>();
    OUTPUT_SOCKETS outputs = data[ROLE_OUTPUTS].value<OUTPUT_SOCKETS>();

    ret.nodeParams->setInputSockets(inputs);
    ret.nodeParams->setParams(params);
    ret.nodeParams->setOutputSockets(outputs);

    VPARAM_INFO panelInfo;
    if (data.find(ROLE_CUSTOMUI_PANEL_IO) != data.end())
    {
        panelInfo = data[ROLE_CUSTOMUI_PANEL_IO].value<VPARAM_INFO>();
    }
    ret.panelParams = new PanelParamModel(ret.nodeParams, panelInfo, nodeIdx, m_pGraphsModel, this);
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

    if (item.panelParams)
    {
        item.panelParams->clear();
        delete item.panelParams;
        item.panelParams = nullptr;
    }
    if (item.nodeParams)
    {
        item.nodeParams->clearParams();
        delete item.nodeParams;
        item.nodeParams = nullptr;
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

QModelIndex SubGraphModel::nodeParamIndex(const QModelIndex &nodeIdx, PARAM_CLASS cls, const QString &paramName) const
{
    _NodeItem item;
    if (!itemFromIndex(nodeIdx, item))
        return QModelIndex();
    return item.nodeParams->getParam(cls, paramName);
    }

ViewParamModel* SubGraphModel::viewParams(const QModelIndex& index)
{
    _NodeItem item;
    if (!itemFromIndex(index, item))
        return nullptr;
    return item.panelParams;
}

ViewParamModel* SubGraphModel::nodeParams(const QModelIndex& index)
{
    _NodeItem item;
    if (!itemFromIndex(index, item))
        return nullptr;
    return item.nodeParams;
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
        case ROLE_OBJDATA:  return QVariant::fromValue(nodeData(index));
        case ROLE_NODETYPE: return item.type;
        case ROLE_INPUTS:
        {
            //legacy interface.
            if (!item.nodeParams)
                return QVariant();

            INPUT_SOCKETS inputs;
            item.nodeParams->getInputSockets(inputs);
            return QVariant::fromValue(inputs);
        }
        case ROLE_OUTPUTS:
        {
            if (!item.nodeParams)
                return QVariant();

            OUTPUT_SOCKETS outputs;
            item.nodeParams->getOutputSockets(outputs);
            return QVariant::fromValue(outputs);
        }
        case ROLE_PARAMETERS:
        {
            if (!item.nodeParams)
                return QVariant();

            PARAMS_INFO params;
            item.nodeParams->getParams(params);
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
        case ROLE_INPUT_MODEL:
        {
            return QVariant();
        }
        case ROLE_PARAM_MODEL:
        {
            return QVariant();
        }
        case ROLE_OUTPUT_MODEL:
        {
            return QVariant();
        }
        case ROLE_PANEL_PARAMS:
        {
            return QVariantPtr<ViewParamModel>::asVariant(item.panelParams);
        }
        case ROLE_NODE_PARAMS:
        {
            return QVariantPtr<QStandardItemModel>::asVariant(item.nodeParams);
        }
        case ROLE_OBJPATH:
        {
            const QModelIndex& subgIdx = m_pGraphsModel->indexBySubModel(const_cast<SubGraphModel*>(this));
            const QString& subgPath = subgIdx.data(ROLE_OBJPATH).toString();
            const QString& path = subgPath + cPathSeperator + item.objid;
            return path;
        }
        case ROLE_CUSTOMUI_PANEL_IO:
        {
            VPARAM_INFO root = item.panelParams->exportParams();
            return QVariant::fromValue(root);
        }
        case ROLE_PARAMS_NO_DESC: 
        {
            return QVariant::fromValue(item.paramNotDesc);
        }
        case ROLE_SUBGRAPH_IDX:
        {
            const QModelIndex& subgIdx = m_pGraphsModel->indexBySubModel(const_cast<SubGraphModel*>(this));
            return subgIdx;
        }
        default:
            return QVariant();
    }
    return QVariant();
}

bool SubGraphModel::setData(const QModelIndex& index, const QVariant& value, int role)
{
    QString id = m_row2Key[index.row()];
    if (m_nodes.find(id) != m_nodes.end())
    {
        _NodeItem& item = m_nodes[id];

        switch (role)
        {
            case ROLE_OBJNAME: 
            {
                item.objCls = value.toString();
                break;
            }
            case ROLE_INPUTS:
            {
                INPUT_SOCKETS inputs = value.value<INPUT_SOCKETS>();
                if (inputs.empty())
                    return false;

                ZASSERT_EXIT(item.nodeParams, false);
                for (QString name : inputs.keys())
                {
                    const INPUT_SOCKET& inSocket = inputs[name];
                    item.nodeParams->setAddParam(
                                PARAM_INPUT, 
                                name,
                                inSocket.info.type,
                                inSocket.info.defaultValue,
                                inSocket.info.control,
                                inSocket.info.ctrlProps,
                                (SOCKET_PROPERTY)inSocket.info.sockProp,
                                inSocket.info.dictpanel,
                                inSocket.info.toolTip
                    );
                }
                break;
            }
            case ROLE_OUTPUTS:
            {
                OUTPUT_SOCKETS outputs = value.value<OUTPUT_SOCKETS>();
                if (outputs.empty())
                    return false;

                ZASSERT_EXIT(item.nodeParams, false);
                for (QString name : outputs.keys())
                {
                    const OUTPUT_SOCKET& outSocket = outputs[name];
                    item.nodeParams->setAddParam(
                                PARAM_OUTPUT,
                                name,
                                outSocket.info.type,
                                outSocket.info.defaultValue,
                                outSocket.info.control,
                                outSocket.info.ctrlProps,
                                (SOCKET_PROPERTY)outSocket.info.sockProp, 
                                outSocket.info.dictpanel,
                                outSocket.info.toolTip
                    );
                    }
                break;
            }
            case ROLE_PARAMETERS:
            {
                PARAMS_INFO params = value.value<PARAMS_INFO>();
                if (params.empty())
                    return false;

                ZASSERT_EXIT(item.nodeParams, false);
                for (QString name : params.keys())
                {
                    const PARAM_INFO& param = params[name];
                    item.nodeParams->setAddParam(
                                PARAM_PARAM,
                                name,
                                param.typeDesc,
                                param.value,
                                param.control,
                                param.controlProps,
                                SOCKPROP_UNKNOWN,
                                DICTPANEL_INFO(),
                                param.toolTip);
                    }
                break;
            }
            case ROLE_CUSTOMUI_PANEL_IO:
            {
                const VPARAM_INFO& invisibleRoot = value.value<VPARAM_INFO>();
                ZASSERT_EXIT(item.panelParams, false);
                item.panelParams->importPanelParam(invisibleRoot);
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
                #if 0
                qDebug() << id << item.viewpos;     //Debug item pos.
                #endif
                break;
            }
            case ROLE_PARAMS_NO_DESC: {
                item.paramNotDesc = value.value<PARAMS_INFO>();
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

bool SubGraphModel::setParamValue(
        PARAM_CLASS cls,
        const QModelIndex& idx,
        const QString& sockName,
        const QVariant& value,
        const QString& type,
        PARAM_CONTROL ctrl,
        SOCKET_PROPERTY sockProp)
{
    _NodeItem item;
    if (!itemFromIndex(idx, item))
        return false;

    //todo: control properties
    item.nodeParams->setAddParam(cls, sockName, type, value, ctrl, QVariant(), sockProp);
    return false;
}

SubGraphModel* SubGraphModel::clone(GraphsModel* parent)
{
    SubGraphModel *pClone = new SubGraphModel(*this);
    pClone->m_pGraphsModel = parent;
    return pClone;
}

QVariant SubGraphModel::getParamValue(const QString& nodeid, const QString& paramName)
{
    if (m_nodes.find(nodeid) == m_nodes.end())
        return QVariant();
    return m_nodes[nodeid].nodeParams->getValue(PARAM_PARAM, paramName);
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

bool SubGraphModel::_insertNode(int row, const NODE_DATA& nodeData, const QModelIndex &parent)
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

    QUuid uuid = QUuid::createUuid();
    uint32_t ident = uuid.data1;
    m_num2strId[ident] = id;
    m_str2numId[id] = ident;

    _NodeItem& item = m_nodes[id];
    QModelIndex nodeIdx = index(row, 0, QModelIndex());
    QModelIndex subgIdx = m_pGraphsModel->indexBySubModel(this);

    importNodeItem(nodeData, nodeIdx, item);

    m_pGraphsModel->markDirty();
    return true;
}

bool SubGraphModel::insertRow(int row, const NODE_DATA &nodeData, const QModelIndex &parent)
{
    beginInsertRows(QModelIndex(), row, row);
    bool ret = _insertNode(row, nodeData);
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
