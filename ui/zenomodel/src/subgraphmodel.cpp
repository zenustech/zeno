#include "graphsmodel.h"
#include "subgraphmodel.h"
#include "modelrole.h"
#include "modeldata.h"
#include <zeno/utils/log.h>
#include "uihelper.h"
#include "zassert.h"
#include "variantptr.h"


SubGraphModel::SubGraphModel(IGraphsModel* pGraphsModel, QObject *parent)
    : QAbstractItemModel(pGraphsModel)
    , m_pGraphsModel(pGraphsModel)
    , m_stack(new QUndoStack(this))
{
    connect(this, &QAbstractItemModel::dataChanged, this,
            [=](const QModelIndex &topLeft, const QModelIndex &bottomRight, const QVector<int> &roles)
    {
        if (m_pGraphsModel) {
            const QModelIndex& subgIdx = m_pGraphsModel->index(m_name);
            emit m_pGraphsModel->_dataChanged(subgIdx, topLeft, roles[0]);
        }
    });

    connect(this, &QAbstractItemModel::rowsAboutToBeInserted, m_pGraphsModel,
            [=](const QModelIndex &parent, int first, int last) {
            if (m_pGraphsModel) {
                const QModelIndex& subgIdx = m_pGraphsModel->index(m_name);
                emit m_pGraphsModel->_rowsAboutToBeInserted(subgIdx, first, last);
            }
    });

    connect(this, &QAbstractItemModel::rowsInserted, m_pGraphsModel, 
            [=](const QModelIndex &parent, int first, int last) {
            if (m_pGraphsModel) {
                const QModelIndex& subgIdx = m_pGraphsModel->index(m_name);
                emit m_pGraphsModel->_rowsInserted(subgIdx, parent, first, last);
            }
    });

    connect(this, &QAbstractItemModel::rowsAboutToBeRemoved, m_pGraphsModel,
            [=](const QModelIndex &parent, int first, int last) {
            if (m_pGraphsModel) {
                const QModelIndex& subgIdx = m_pGraphsModel->index(m_name);
                emit m_pGraphsModel->_rowsAboutToBeRemoved(subgIdx, parent, first, last);
            }
    });

    connect(this, &QAbstractItemModel::rowsRemoved, m_pGraphsModel,
            [=](const QModelIndex &parent, int first, int last) {
            if (m_pGraphsModel) {
                const QModelIndex& subgIdx = m_pGraphsModel->index(m_name);
                emit m_pGraphsModel->_rowsRemoved(parent, first, last);
            }
    });
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

bool SubGraphModel::checkCustomName(const QString &name) 
{
    if (name.isEmpty())
        return true;
    for (auto node : m_nodes) {
        if (node->customName == name)
            return false;
    }
    return true;
}

void SubGraphModel::clear()
{
    m_nodes.clear();
    m_key2Row.clear();
    m_row2Key.clear();
    m_name2identLst.clear();
    m_num2strId.clear();
    m_str2numId.clear();

    const QModelIndex& subgIdx = m_pGraphsModel->index(m_name);
    emit m_pGraphsModel->clearLayout(subgIdx);
}

NODE_DATA SubGraphModel::item2NodeData(const NodeItem* item) const
{
    NODE_DATA data;
    data.ident = item->objid;
    data.nodeCls = item->objCls;
    data.customName = item->customName;
    data.pos = item->viewpos;
    data.bCollasped = item->bCollasped;
    data.options = item->options;
    data.type = item->type;

    INPUT_SOCKETS inputs;
    OUTPUT_SOCKETS outputs;
    PARAMS_INFO params;

    item->nodeParams->getInputSockets(data.inputs);
    item->nodeParams->getParams(data.params);
    item->nodeParams->getOutputSockets(data.outputs);

    data.customPanel = item->panelParams->exportParams();
    data.parmsNotDesc = item->paramNotDesc;

    return data;
}

NodeItem* SubGraphModel::importNodeItem(const NODE_DATA& data)
{
    NodeItem* pItem = new NodeItem(this);

    pItem->objid = data.ident;
    pItem->objCls = data.nodeCls;
    pItem->customName = data.customName;
    pItem->viewpos = data.pos;
    pItem->bCollasped = data.bCollasped;
    pItem->options = data.options;
    pItem->type = data.type;
    pItem->paramNotDesc = data.parmsNotDesc;

    const QModelIndex& subgIdx = m_pGraphsModel->index(m_name);

    pItem->nodeParams = new NodeParamModel(m_pGraphsModel, false, pItem);

    INPUT_SOCKETS inputs = data.inputs;
    PARAMS_INFO params = data.params;
    OUTPUT_SOCKETS outputs = data.outputs;

    pItem->nodeParams->setInputSockets(inputs);
    pItem->nodeParams->setParams(params);
    pItem->nodeParams->setOutputSockets(outputs);
    pItem->panelParams = new PanelParamModel(pItem->nodeParams, data.customPanel, m_pGraphsModel, pItem);

    return pItem;
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

QModelIndex SubGraphModel::index(uint32_t id) const
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
    NodeItem* item = itemFromIndex(index);
    if (!item)
        return false;

    QString currNode = index.data(ROLE_OBJID).toString();
    const QModelIndex& subgIdx = m_pGraphsModel->index(m_name);

    if (item->panelParams)
    {
        item->panelParams->clear();
        delete item->panelParams;
        item->panelParams = nullptr;
    }
    if (item->nodeParams)
    {
        item->nodeParams->clearParams();
        delete item->nodeParams;
        item->nodeParams = nullptr;
    }

    int row = index.row();
    QString id = m_row2Key[row];
    QString name = m_nodes[id]->objCls;
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
    m_name2identLst[name].remove(id);

    uint32_t numId = m_str2numId[id];
    m_num2strId.remove(numId);
    m_str2numId.remove(id);

    m_pGraphsModel->markDirty();
    return true;
}

void SubGraphModel::_removeNetLabels(const NodeParamModel* nodeParams)
{
    for (const auto& inputSocketIdx : nodeParams->getInputIndice())
    {
        QString netLabel = inputSocketIdx.data(ROLE_PARAM_NETLABEL).toString();
        if (!netLabel.isEmpty())
        {
            removeNetLabel(inputSocketIdx, netLabel);
        }
        if (QAbstractItemModel* pKeyObjModel =
            QVariantPtr<QAbstractItemModel>::asPtr(inputSocketIdx.data(ROLE_VPARAM_LINK_MODEL)))
        {
            for (int i = 0; i < pKeyObjModel->rowCount(); i++)
            {
                const QModelIndex& idx = pKeyObjModel->index(i, 0);
                netLabel = idx.data(ROLE_PARAM_NETLABEL).toString();
                if (!netLabel.isEmpty())
                {
                    removeNetLabel(idx, netLabel);
                }
            }
        }
    }
    for (const auto& outputSocketIdx : nodeParams->getOutputIndice())
    {
        QString netLabel = outputSocketIdx.data(ROLE_PARAM_NETLABEL).toString();
        if (!netLabel.isEmpty())
        {
            removeNetLabel(outputSocketIdx, netLabel);
        }
        if (QAbstractItemModel* pKeyObjModel =
            QVariantPtr<QAbstractItemModel>::asPtr(outputSocketIdx.data(ROLE_VPARAM_LINK_MODEL)))
        {
            for (int i = 0; i < pKeyObjModel->rowCount(); i++)
            {
                const QModelIndex& idx = pKeyObjModel->index(i, 0);
                netLabel = idx.data(ROLE_PARAM_NETLABEL).toString();
                if (!netLabel.isEmpty())
                {
                    removeNetLabel(idx, netLabel);
                }
            }
        }
    }
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
    NodeItem* item = itemFromIndex(nodeIdx);
    if (!item)
        return QModelIndex();
    return item->nodeParams->getParam(cls, paramName);
}

ViewParamModel* SubGraphModel::viewParams(const QModelIndex& index)
{
    NodeItem* item = itemFromIndex(index);
    if (!item)
        return nullptr;
    return item->panelParams;
}

ViewParamModel* SubGraphModel::nodeParams(const QModelIndex& index)
{
    NodeItem* item = itemFromIndex(index);
    if (!item)
        return nullptr;
    return item->nodeParams;
}

QVariant SubGraphModel::data(const QModelIndex& index, int role) const
{
    NodeItem* item = itemFromIndex(index);
    if (!item)
        return QVariant();

    switch (role)
    {
        case ROLE_OBJID:    return item->objid;
        case ROLE_OBJNAME:  return item->objCls;
        case ROLE_CUSTOM_OBJNAME: return item->customName;
        case ROLE_OBJDATA:  return QVariant::fromValue(nodeData(index));
        case ROLE_NODETYPE: return item->type;
        case ROLE_INPUTS:
        {
            //legacy interface.
            if (!item->nodeParams)
                return QVariant();

            INPUT_SOCKETS inputs;
            item->nodeParams->getInputSockets(inputs);
            return QVariant::fromValue(inputs);
        }
        case ROLE_OUTPUTS:
        {
            if (!item->nodeParams)
                return QVariant();

            OUTPUT_SOCKETS outputs;
            item->nodeParams->getOutputSockets(outputs);
            return QVariant::fromValue(outputs);
        }
        case ROLE_PARAMETERS:
        {
            if (!item->nodeParams)
                return QVariant();

            PARAMS_INFO params;
            item->nodeParams->getParams(params);
            return QVariant::fromValue(params);
        }
        case ROLE_COLLASPED:
        {
            return item->bCollasped;
        }
        case ROLE_NODE_DATACHANGED:
        {
            return false;
        }
        case ROLE_OPTIONS:
        {
            return item->options;
        }
        case ROLE_OBJPOS:
        {
            return item->viewpos;
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
            return QVariantPtr<ViewParamModel>::asVariant(item->panelParams);
        }
        case ROLE_NODE_PARAMS:
        {
            return QVariantPtr<QStandardItemModel>::asVariant(item->nodeParams);
        }
        case ROLE_OBJPATH:
        {
            const QModelIndex &subgIdx = m_pGraphsModel->index(m_name);
            const QString& subgPath = subgIdx.data(ROLE_OBJPATH).toString();
            const QString& path = subgPath + "/" + item->objid;
            return path;
        }
        case ROLE_CUSTOMUI_PANEL_IO:
        {
            VPARAM_INFO root = item->panelParams->exportParams();
            return QVariant::fromValue(root);
        }
        case ROLE_PARAMS_NO_DESC: 
        {
            return QVariant::fromValue(item->paramNotDesc);
        }
        case ROLE_SUBGRAPH_IDX:
        {
            const QModelIndex &subgIdx = m_pGraphsModel->index(m_name);
            return subgIdx;
        }
        case ROLE_KEYFRAMES: {
            if (!item.nodeParams)
                return QVariant();
            QVector<int> keys;
            for (const QModelIndex &index : item.nodeParams->getInputIndice()) {
                QVariant value = index.data(ROLE_PARAM_VALUE);
                int ctrl = index.data(ROLE_PARAM_CTRL).toInt();
                if (value.canConvert<CURVES_DATA>() && ctrl != CONTROL_CURVE) {
                    CURVES_DATA curves = value.value<CURVES_DATA>();
                    for (CURVE_DATA &curve : curves)
                    {
                        keys << curve.pointBases();
                    }
                }
            }
            keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
            return QVariant::fromValue(keys);
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
        auto item = m_nodes[id];

        switch (role)
        {
            case ROLE_OBJNAME: 
            {
                item->objCls = value.toString();
                break;
            }
            case ROLE_CUSTOM_OBJNAME: 
            {
                bool isValid = checkCustomName(value.toString());
                if (isValid)
                    item->customName = value.toString();
                else
                    return isValid;
                break;
            }
            case ROLE_INPUTS:
            {
                INPUT_SOCKETS inputs = value.value<INPUT_SOCKETS>();
                if (inputs.empty())
                    return false;

                ZASSERT_EXIT(item->nodeParams, false);
                for (QString name : inputs.keys())
                {
                    const INPUT_SOCKET& inSocket = inputs[name];
                    item->nodeParams->setAddParam(
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

                ZASSERT_EXIT(item->nodeParams, false);
                for (QString name : outputs.keys())
                {
                    const OUTPUT_SOCKET& outSocket = outputs[name];
                    item->nodeParams->setAddParam(
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

                ZASSERT_EXIT(item->nodeParams, false);
                for (QString name : params.keys())
                {
                    const PARAM_INFO& param = params[name];
                    item->nodeParams->setAddParam(
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
                ZASSERT_EXIT(item->panelParams, false);
                item->panelParams->importPanelParam(invisibleRoot);
                break;
            }
            case ROLE_COLLASPED:
            {
                item->bCollasped = value.toBool();
                break;
            }
            case ROLE_OPTIONS:
            {
                item->options = value.toInt();
                break;
            }
            case ROLE_NODE_DATACHANGED:
            {
                break;
            }
            case ROLE_OBJPOS:
            {
                item->viewpos = value.toPointF();
                #if 0
                qDebug() << id << item->viewpos;     //Debug item pos.
                #endif
                break;
            }
            case ROLE_PARAMS_NO_DESC: {
                item->paramNotDesc = value.value<PARAMS_INFO>();
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
    NodeItem* item = itemFromIndex(idx);
    if (!item)
        return false;

    //todo: control properties
    item->nodeParams->setAddParam(cls, sockName, type, value, ctrl, QVariant(), sockProp);
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
    return m_nodes[nodeid]->nodeParams->getValue(PARAM_PARAM, paramName);
}

void SubGraphModel::updateNodeStatus(const QString& nodeid, STATUS_UPDATE_INFO info)
{
    auto it = m_nodes.find(nodeid);
    if (it == m_nodes.end())
        return;

    auto item = m_nodes[nodeid];
    switch (info.role)
    {
    case ROLE_COLLASPED: item->bCollasped = info.newValue.toBool(); break;
    case ROLE_OPTIONS: item->options = info.newValue.toInt(); break;
    case ROLE_OBJPOS: item->viewpos = info.newValue.toPointF(); break;
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

NODE_DATA SubGraphModel::nodeData(const QModelIndex &idx) const
{
    if (NodeItem* item = itemFromIndex(idx))
    {
        return item2NodeData(item);
    }
    return NODE_DATA();
}

QModelIndexList SubGraphModel::match(const QModelIndex& start, int role, const QVariant& value, int hits, Qt::MatchFlags flags) const
{
    return _base::match(start, role, value, hits, flags);
}

QModelIndexList SubGraphModel::getNodesByCls(const QString& nodeCls)
{
    QModelIndexList nodes;
    auto iter = m_name2identLst.find(nodeCls);
    if (iter != m_name2identLst.end())
    {
        for (QString ident : iter.value())
        {
            QModelIndex idx = index(ident);
            nodes.append(idx);
        }
    }
    return nodes;
}

NodeItem* SubGraphModel::itemFromIndex(const QModelIndex &index) const
{
    if (!index.isValid())
        return nullptr;

    auto iter1 = m_row2Key.find(index.row());
    if (iter1 == m_row2Key.end())
        return nullptr;

    QString id = iter1.value();
    auto iter2 = m_nodes.find(id);
    if (iter2 != m_nodes.end())
    {
        return iter2.value();
    }
    else
    {
        return nullptr;
    }
}

bool SubGraphModel::_insertNode(int row, const NODE_DATA& nodeData, const QModelIndex &parent)
{
    //pure insert logic, without transaction and notify stuffs.
    const QString& id = nodeData.ident;
    const QString& name = nodeData.nodeCls;

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

    m_name2identLst[name].insert(id);

    NodeItem *pItem = importNodeItem(nodeData);
    m_nodes.insert(id, pItem);
    const QModelIndex& subgIdx = m_pGraphsModel->index(m_name);

    if (nodeData.find(ROLE_NODETYPE) != nodeData.end() &&
        nodeData[ROLE_NODETYPE] == NO_VERSION_NODE)
    {
        m_pGraphsModel->markNotDescNode();
    }

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
    auto iter = m_name2identLst.find(oldName);
    if (iter == m_name2identLst.end())
        return;

    auto sets = iter.value();
    for (QString ident : sets)
    {
        QModelIndex idx = index(ident);
        setData(idx, newName, ROLE_OBJNAME);
    }

    m_name2identLst.remove(oldName);
    m_name2identLst.insert(newName, sets);
}

bool SubGraphModel::addNetLabel(const QModelIndex& sock, const QString& name, bool bInput)
{
    int nRow = m_labels.size();
    _LabelItem& item = m_labels[name];
    item.name = name;
    if (bInput)
        item.inSocks.append(sock);
    else
        item.outSock = sock;

    auto paramModel = const_cast<QAbstractItemModel*>(sock.model());
    ZASSERT_EXIT(paramModel, false);
    paramModel->setData(sock, name, ROLE_PARAM_NETLABEL);
    return true;
}

void SubGraphModel::updateNetLabel(const QModelIndex& trigger, const QString& oldName, const QString& newName)
{
    if (oldName == newName
        || m_labels.find(oldName) == m_labels.end() 
        || m_labels.find(newName) != m_labels.end())
        return;

    auto item = m_labels[oldName];
    for (QPersistentModelIndex insock : item.inSocks)
    {
        auto pModel = const_cast<QAbstractItemModel*>(insock.model());
        pModel->setData(insock, newName, ROLE_PARAM_NETLABEL);
    }
    auto pModel = const_cast<QAbstractItemModel*>(item.outSock.model());
    pModel->setData(item.outSock, newName, ROLE_PARAM_NETLABEL);

    if (newName.isEmpty()) {
        m_labels.remove(oldName);
    }
    else {
        item.name = newName;
        m_labels.insert(newName, item);
        m_labels.remove(oldName);
    }
}

void SubGraphModel::removeNetLabel(const QModelIndex& trigger, const QString& name)
{
    if (!trigger.isValid())
        return;
    auto pModel = const_cast<QAbstractItemModel*>(trigger.model());
    pModel->setData(trigger, "", ROLE_PARAM_NETLABEL);
    auto iter = m_labels.find(name);
    if (iter != m_labels.end())
    {
        auto& item = iter.value();
        if (item.outSock == trigger) {
            for (auto inSocket : item.inSocks)
            {
                if (!inSocket.isValid())
                    continue;
                if (auto pInSocketModel = const_cast<QAbstractItemModel*>(inSocket.model()))
                    pInSocketModel->setData(inSocket, "", ROLE_PARAM_NETLABEL);
            }
            m_labels.remove(name);
        }
        else if (item.inSocks.indexOf(trigger) != -1) {
            item.inSocks.removeAll(trigger);
        }
    }
}

QModelIndex SubGraphModel::getNetOutput(const QString& name) const
{
    auto iter = m_labels.find(name);
    if (iter == m_labels.end())
        return QModelIndex();
    return iter.value().outSock;
}

QModelIndexList SubGraphModel::getNetInputSocks(const QString& name) const
{
    auto iter = m_labels.find(name);
    if (iter == m_labels.end())
        return QList<QModelIndex>();
    QModelIndexList inSocks;
    for (auto inSock : iter.value().inSocks)
        inSocks.append(inSock);
    return inSocks;
}

QStringList SubGraphModel::dumpLabels() const
{
    QStringList names;
    for (auto iter : m_labels)
    {
        if (!iter.name.isEmpty())
            names.append(iter.name);
    }
    return names;
}

