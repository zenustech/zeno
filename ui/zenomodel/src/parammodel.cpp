#include "parammodel.h"
#include "zassert.h"
#include "igraphsmodel.h"
#include "linkmodel.h"
#include "variantptr.h"
#include "globalcontrolmgr.h"
#include "dictkeymodel.h"
#include <zenomodel/include/uihelper.h>
#include <zeno/utils/scope_exit.h>


IParamModel::IParamModel(
        PARAM_CLASS paramType,
        IGraphsModel* pModel,
        const QPersistentModelIndex& subgIdx,
        const QPersistentModelIndex& nodeIdx,
        QObject* parent)
    : QAbstractItemModel(parent)
    , m_nodeIdx(nodeIdx)
    , m_subgIdx(subgIdx)
    , m_model(pModel)
    , m_class(paramType)
    , m_bRetryLinkOp(false)
{
    Q_ASSERT(m_model);
}

IParamModel::~IParamModel()
{
}

EdgeInfo IParamModel::exportLink(const QModelIndex& linkIdx)
{
    EdgeInfo link;

    QModelIndex outSock = linkIdx.data(ROLE_OUTSOCK_IDX).toModelIndex();
    QModelIndex inSock = linkIdx.data(ROLE_INSOCK_IDX).toModelIndex();
    ZASSERT_EXIT(outSock.isValid() && inSock.isValid(), link);

    QModelIndex outCoreParam = outSock.data(ROLE_PARAM_COREIDX).toModelIndex();
    QModelIndex inCoreParam = inSock.data(ROLE_PARAM_COREIDX).toModelIndex();
    ZASSERT_EXIT(outCoreParam.isValid() && inCoreParam.isValid(), link);

    //for dict panel socket, write the full path of output socket.
    if (outSock.data(ROLE_PARAM_CLASS) == PARAM_INNER_OUTPUT)
    {
        link.outSockPath = outSock.data(ROLE_OBJPATH).toString();
    }
    else
    {
        link.outSockPath = outCoreParam.data(ROLE_OBJPATH).toString();
    }

    if (inSock.data(ROLE_PARAM_CLASS) == PARAM_INNER_INPUT)
    {
        link.inSockPath = inSock.data(ROLE_OBJPATH).toString();
    }
    else
    {
        link.inSockPath = inCoreParam.data(ROLE_OBJPATH).toString();
    }
    return link;
}

void IParamModel::exportDictkeys(DictKeyModel *pModel, DICTPANEL_INFO& panel)
{
    if (!pModel)
        return;

    panel.bCollasped = pModel->isCollasped();

    int rowCnt = pModel->rowCount();
    QStringList keyNames;
    for (int i = 0; i < rowCnt; i++)
    {
        const QModelIndex &keyIdx = pModel->index(i, 0);
        QString key = keyIdx.data().toString();

        DICTKEY_INFO keyInfo;
        keyInfo.key = key;

        PARAM_LINKS links = keyIdx.data(ROLE_PARAM_LINKS).value<PARAM_LINKS>();
        for (auto linkIdx : links)
        {
            if (linkIdx.isValid())
            {
                QModelIndex outsock = linkIdx.data(ROLE_OUTSOCK_IDX).toModelIndex();
                QModelIndex insock = linkIdx.data(ROLE_INSOCK_IDX).toModelIndex();
                ZASSERT_EXIT(insock.isValid() && outsock.isValid());

                EdgeInfo link = exportLink(linkIdx);
                keyInfo.links.append(link);
            }
        }

        panel.keys.append(keyInfo);
        keyNames.push_back(key);
    }
}

QList<EdgeInfo> IParamModel::exportLinks(const PARAM_LINKS& links)
{
    QList<EdgeInfo> linkInfos;
    for (auto linkIdx : links)
    {
        EdgeInfo link = exportLink(linkIdx);
        linkInfos.append(link);
    }
    return linkInfos;
}

bool IParamModel::getInputSockets(INPUT_SOCKETS& inputs)
{
    if (m_class == PARAM_INPUT)
    {
        for (int r = 0; r < rowCount(); r++)
        {
            QModelIndex idx = index(r, 0);

            QString name = nameFromRow(idx.row());
            auto itItem = m_items.find(name);
            ZASSERT_EXIT(itItem != m_items.end(), false);
            _ItemInfo& item = m_items[name];

            INPUT_SOCKET inSocket;
            inSocket.info.defaultValue = item.pConst;
            inSocket.info.nodeid = m_nodeIdx.data(ROLE_OBJID).toString();
            inSocket.info.name = name;
            inSocket.info.type = item.type;
            inSocket.info.sockProp = item.prop;
            inSocket.info.links = exportLinks(item.links);

            if (item.customData.find(ROLE_VPARAM_LINK_MODEL) != item.customData.end())
            {
                DictKeyModel *pModel = QVariantPtr<DictKeyModel>::asPtr(item.customData[ROLE_VPARAM_LINK_MODEL]);
                ZASSERT_EXIT(pModel, false);
                exportDictkeys(pModel, inSocket.info.dictpanel);
            }
            inputs.insert(name, inSocket);
        }
        return true;
    }
    return false;
}

bool IParamModel::getOutputSockets(OUTPUT_SOCKETS& outputs)
{
    if (m_class != PARAM_OUTPUT)
        return false;

    for (int r = 0; r < rowCount(); r++)
    {
        QModelIndex idx = index(r, 0);

        QString name = nameFromRow(idx.row());
        auto itItem = m_items.find(name);
        ZASSERT_EXIT(itItem != m_items.end(), false);
        _ItemInfo& item = m_items[name];

        {
            OUTPUT_SOCKET outSocket;
            outSocket.info.defaultValue = item.pConst;
            outSocket.info.nodeid = m_nodeIdx.data(ROLE_OBJID).toString();
            outSocket.info.name = name;
            outSocket.info.type = item.type;
            outSocket.info.sockProp = item.prop;
            outSocket.info.links = exportLinks(item.links);

            if (item.customData.find(ROLE_VPARAM_LINK_MODEL) != item.customData.end()) {
                DictKeyModel *pModel = QVariantPtr<DictKeyModel>::asPtr(item.customData[ROLE_VPARAM_LINK_MODEL]);
                ZASSERT_EXIT(pModel, false);
                exportDictkeys(pModel, outSocket.info.dictpanel);
            }

            outputs.insert(name, outSocket);
        }
    }
    return true;
}

bool IParamModel::getParams(PARAMS_INFO& params)
{
    if (m_class != PARAM_PARAM)
        return false;

    for (int r = 0; r < rowCount(); r++)
    {
        QModelIndex idx = index(r, 0);

        QString name = nameFromRow(idx.row());
        auto itItem = m_items.find(name);
        ZASSERT_EXIT(itItem != m_items.end(), false);
        _ItemInfo& item = m_items[name];

        {
            PARAM_INFO paramInfo;
            paramInfo.bEnableConnect = false;
            paramInfo.value = item.pConst;
            paramInfo.typeDesc = item.type;
            paramInfo.name = name;
            params.insert(name, paramInfo);
        }
    }
    return true;
}

void IParamModel::setInputSockets(const INPUT_SOCKETS& inputs)
{
    for (INPUT_SOCKET inSocket : inputs)
    {
        _ItemInfo item;
        item.name = inSocket.info.name;
        item.pConst = inSocket.info.defaultValue;
        item.type = inSocket.info.type;
        item.prop = inSocket.info.sockProp;
        appendRow(item.name, item.type, item.pConst, item.prop);
    }
}

void IParamModel::setParams(const PARAMS_INFO& params)
{
    for (PARAM_INFO paramInfo : params)
    {
        _ItemInfo item;
        item.name = paramInfo.name;
        item.pConst = paramInfo.value;
        item.type = paramInfo.typeDesc;
        item.prop = SOCKPROP_UNKNOWN;
        appendRow(item.name, item.type, item.pConst, item.prop);
    }
}

void IParamModel::setOutputSockets(const OUTPUT_SOCKETS& outputs)
{
    for (OUTPUT_SOCKET outSocket : outputs)
    {
        _ItemInfo item;
        item.name = outSocket.info.name;
        item.pConst = outSocket.info.defaultValue;
        item.type = outSocket.info.type;
        item.prop = outSocket.info.sockProp;
        appendRow(item.name, item.type, item.pConst, item.prop);
    }
}

PARAM_CLASS IParamModel::paramClass() const
{
    return m_class;
}

QModelIndex IParamModel::index(int row, int column, const QModelIndex& parent) const
{
    if (row < 0 || row >= rowCount())
        return QModelIndex();

    auto itRow = m_row2Key.find(row);
    ZASSERT_EXIT(itRow != m_row2Key.end(), QModelIndex());

    QString name = itRow.value();
    auto itItem = m_items.find(name);
    ZASSERT_EXIT(itItem != m_items.end(), QModelIndex());

    return createIndex(row, 0, nullptr);
}

QModelIndex IParamModel::index(const QString& name) const
{
    //may be a dict-key name


    if (m_key2Row.find(name) == m_key2Row.end())
        return QModelIndex();

    int row = m_key2Row[name];
    return createIndex(row, 0, nullptr);
}

QString IParamModel::nameFromRow(int row) const
{
    QString name;
    auto itRow = m_row2Key.find(row);
    ZASSERT_EXIT(itRow != m_row2Key.end(), name);
    name = itRow.value();
    return name;
}

void IParamModel::clear()
{
    while (rowCount() > 0)
    {
        //safe to notify the remove msg.
        removeRows(0, 1);
    }
}

QModelIndex IParamModel::parent(const QModelIndex& child) const
{
    return QModelIndex();
}

int IParamModel::rowCount(const QModelIndex& parent) const
{
    return m_items.size();
}

int IParamModel::columnCount(const QModelIndex& parent) const
{
    return 1;
}

bool IParamModel::hasChildren(const QModelIndex& parent) const
{
    return false;
}

QVariant IParamModel::data(const QModelIndex& index, int role) const
{
    const QString& name = nameFromRow(index.row());
    auto itItem = m_items.find(name);
    ZASSERT_EXIT(itItem != m_items.end(), QModelIndex());
    const _ItemInfo& item = m_items[name];

    switch (role)
    {
        case Qt::DisplayRole:
        case ROLE_PARAM_NAME:   return item.name;
        case ROLE_PARAM_TYPE:   return item.type;
        case ROLE_PARAM_CTRL:   return m_tempControl;    //control belongs to view data, only offer the control used to sync.
        case ROLE_PARAM_VALUE:  return item.pConst;
        case ROLE_PARAM_SOCKPROP: return item.prop;
        case ROLE_PARAM_LINKS:  return QVariant::fromValue(item.links);
        case ROLE_PARAM_CLASS:     return m_class;
        case ROLE_OBJID:
            return m_nodeIdx.isValid() ? m_nodeIdx.data(ROLE_OBJID).toString() : "";
        case ROLE_NODE_IDX:
            return m_nodeIdx;
        case ROLE_PARAM_COREIDX:
            return index;
        case ROLE_OBJPATH:
        {
            QString path;
            path = QString("core-param") + cPathSeperator;
            if (m_class == PARAM_INPUT) {
                path += "/inputs";
            }
            else if (m_class == PARAM_PARAM) {
                path += "/params";
            }
            else if (m_class == PARAM_OUTPUT) {
                path += "/outputs";
            }
            path += "/" + name;
            path = m_nodeIdx.data(ROLE_OBJPATH).toString() + cPathSeperator + path;
            return path;
        }
        case ROLE_VPARAM_LINK_MODEL:
        {
            if (item.customData.find(role) != item.customData.end())
            {
                return item.customData[role];
            }
            break;
        }
    }
    return QVariant();
}

bool IParamModel::setData(const QModelIndex& index, const QVariant& value, int role)
{
    if (!index.isValid())
        return false;

    QString name = nameFromRow(index.row());
    auto itItem = m_items.find(name);
    ZASSERT_EXIT(itItem != m_items.end(), false);
    _ItemInfo& item = m_items[name];
    QVariant oldValue;

    switch (role)
    {
        case ROLE_PARAM_NAME:
        {
            //rename
            QString oldName = item.name;
            QString newName = value.toString();
            oldValue = oldName;
            if (oldName == newName)
                return false;

            int row = m_key2Row[name];
            ZASSERT_EXIT(row == index.row(), false);
            m_key2Row.remove(oldName);
            m_key2Row.insert(newName, row);
            m_row2Key[row] = newName;

            _ItemInfo newItem = item;
            newItem.name = newName;
            m_items.remove(oldName);
            m_items.insert(newName, newItem);

            break;
        }
        case ROLE_PARAM_TYPE:
        {
            if (item.type == value.toString())
                return false;
            oldValue = item.type;
            item.type = value.toString();
            break;
        }
        case ROLE_PARAM_VALUE:
        {
            if (item.pConst == value)
                return false;
            oldValue = item.pConst;
            item.pConst = value;
            onSubIOEdited(oldValue, item);
            break;
        }
        case ROLE_PARAM_LINKS:
        {
            oldValue = QVariant::fromValue(item.links);
            item.links = value.value<PARAM_LINKS>();
            break;
        }
        case ROLE_ADDLINK:
        {
            QPersistentModelIndex linkIdx = value.toPersistentModelIndex();
            ZASSERT_EXIT(linkIdx.isValid(), false);
            item.links.append(linkIdx);

            QString nodeId, nodeCls;
            if (m_nodeIdx.isValid())
            {
                nodeId = m_nodeIdx.data(ROLE_OBJID).toString();
                nodeCls = m_nodeIdx.data(ROLE_OBJNAME).toString();
            }
            if (!m_model->IsIOProcessing() &&
                (nodeCls == "MakeList" || nodeCls == "MakeDict" || nodeCls == "ExtractDict"))
            {
                QStringList lst = sockNames();
                int maxObjId = UiHelper::getMaxObjId(lst);
                if (maxObjId == -1)
                    maxObjId = 0;
                QString maxObjSock = QString("obj%1").arg(maxObjId);
                QString lastKey = lst.last();

                if ((nodeCls == "MakeList" || nodeCls == "MakeDict") && name == lastKey)
                {
                    const QString& newObjName = QString("obj%1").arg(maxObjId + 1);
                    SOCKET_PROPERTY prop = nodeCls == "MakeDict" ? SOCKPROP_EDITABLE : SOCKPROP_NORMAL;
                    appendRow(newObjName, "", QVariant(), prop);
                }
                else if (nodeCls == "ExtractDict" && name == lastKey)
                {
                    const QString& newObjName = QString("obj%1").arg(maxObjId + 1);
                    appendRow(newObjName, "", QVariant(), SOCKPROP_EDITABLE);
                }
            }
            break;
        }
        case ROLE_REMOVELINK:
        {
            QPersistentModelIndex linkIdx = value.toPersistentModelIndex();
            ZASSERT_EXIT(linkIdx.isValid(), false);
            item.links.removeAll(linkIdx);
            break;
        }
        default:
            return false;
    }
    emit dataChanged(index, index, QVector<int>{role});
    return true;
}

void IParamModel::onSubIOEdited(const QVariant& oldValue, const _ItemInfo& item)
{
    if (m_model->IsIOProcessing())
        return;

    const QString& nodeName = m_nodeIdx.data(ROLE_OBJNAME).toString();
    if (nodeName == "SubInput" || nodeName == "SubOutput")
    {
        bool bInput = nodeName == "SubInput";
        const QString& subgName = m_subgIdx.data(ROLE_OBJNAME).toString();
        ZASSERT_EXIT(m_items.find("defl") != m_items.end() &&
                     m_items.find("name") != m_items.end() &&
                     m_items.find("type") != m_items.end());
        const QString& sockName = m_items["name"].pConst.toString();

        if (item.name == "type")
        {
            const QString& newType = item.pConst.toString();
            PARAM_CONTROL newCtrl = UiHelper::getControlByType(newType);
            const QVariant& newValue = UiHelper::initDefaultValue(newType);

            const QModelIndex& idx_defl = index("defl");
            
            _ItemInfo& defl = m_items["defl"];
            defl.type = newType;
            defl.pConst = newValue;
            emit dataChanged(idx_defl, idx_defl, QVector<int>{ROLE_PARAM_TYPE});

            //update desc.
            NODE_DESC desc;
            bool ret = m_model->getDescriptor(subgName, desc);
            ZASSERT_EXIT(ret);
            if (bInput)
            {
                ZASSERT_EXIT(desc.inputs.find(sockName) != desc.inputs.end());
                desc.inputs[sockName].info.type = newType;
                desc.inputs[sockName].info.control = newCtrl;
            }
            else
            {
                ZASSERT_EXIT(desc.outputs.find(sockName) != desc.outputs.end());
                desc.outputs[sockName].info.type = newType;
            }
            m_model->updateSubgDesc(subgName, desc);

            //update to every subgraph node.
            QModelIndexList subgNodes = m_model->findSubgraphNode(subgName);
            for (auto idx : subgNodes)
            {
                // update socket for current subgraph node.
                IParamModel* sockModel = QVariantPtr<IParamModel>::asPtr(idx.data(bInput ? ROLE_INPUT_MODEL : ROLE_OUTPUT_MODEL));
                QModelIndex paramIdx = sockModel->index(sockName);
                sockModel->setData(paramIdx, newType, ROLE_PARAM_TYPE);
            }
        }
        else if (item.name == "name")
        {
            //1.update desc info for the subgraph node.
            const QString& newName = sockName;
            const QString& oldName = oldValue.toString();

            NODE_DESC desc;
            bool ret = m_model->getDescriptor(subgName, desc);
            ZASSERT_EXIT(ret);
            if (bInput)
            {
                desc.inputs[newName].info.name = newName;
                desc.inputs.remove(oldName);
            }
            else
            {
                desc.outputs[newName].info.name = newName;
                desc.outputs.remove(oldName);
            }
            m_model->updateSubgDesc(subgName, desc);

            //2.update all sockets for all subgraph node.
            QModelIndexList subgNodes = m_model->findSubgraphNode(subgName);
            for (auto idx : subgNodes)
            {
                //update socket for current subgraph node.
                IParamModel* sockModel = QVariantPtr<IParamModel>::asPtr(idx.data(bInput ? ROLE_INPUT_MODEL : ROLE_OUTPUT_MODEL));
                QModelIndex paramIdx = sockModel->index(oldName);
                sockModel->setData(paramIdx, newName, ROLE_PARAM_NAME);
            }
        }
        else if (item.name == "defl")
        {
            const QVariant& deflVal = m_items["defl"].pConst;
            NODE_DESC desc;
            bool ret = m_model->getDescriptor(subgName, desc);
            ZASSERT_EXIT(ret);
            if (bInput)
            {
                ZASSERT_EXIT(desc.inputs.find(sockName) != desc.inputs.end());
                desc.inputs[sockName].info.defaultValue = deflVal;
            }
            else
            {
                ZASSERT_EXIT(desc.outputs.find(sockName) != desc.outputs.end());
                desc.outputs[sockName].info.defaultValue = deflVal;
            }
            m_model->updateSubgDesc(subgName, desc);
            //no need to update all subgraph node because it causes disturbance.
        }
    }
}

QModelIndexList IParamModel::match(
    const QModelIndex& start,
    int role,
    const QVariant& value,
    int hits,
    Qt::MatchFlags flags) const
{
    return QModelIndexList();
}

bool IParamModel::removeRows(int row, int count, const QModelIndex& parent)
{
    //todo: clear link here, because link model need valid param index.
    auto itRow = m_row2Key.find(row);
    if (itRow == m_row2Key.end())
        return false;

    QString name = itRow.value();
    auto itItem = m_items.find(name);
    ZASSERT_EXIT(itItem != m_items.end(), false);
    PARAM_LINKS socketLinks = m_items[name].links;
    for (const QPersistentModelIndex& linkIdx : socketLinks)
    {
        m_model->removeLink(linkIdx, true);
    }

    beginRemoveRows(parent, row, row);
    _removeRow(index(row, 0));
    endRemoveRows();
    return true;
}

bool IParamModel::_removeRow(const QModelIndex& index)
{
    //remove link from this param.
    QString name = nameFromRow(index.row());

    auto itItem = m_items.find(name);
    ZASSERT_EXIT(itItem != m_items.end(), false);

    int row = index.row();
    for (int r = row + 1; r < rowCount(); r++)
    {
        const QString& key = m_row2Key[r];
        m_row2Key[r - 1] = key;
        m_key2Row[key] = r - 1;
    }
    m_row2Key.remove(rowCount() - 1);
    m_key2Row.remove(name);
    m_items.remove(name);

    m_model->markDirty();
    return true;
}

void IParamModel::insertRow(int row, const QString& sockName, const QString& type, const QVariant& deflValue, int prop)
{
    beginInsertRows(QModelIndex(), row, row);
    bool ret = _insertRow(row, sockName, type, deflValue, prop);
    endInsertRows();
}

void IParamModel::appendRow(const QString& sockName, const QString& type, const QVariant& deflValue, int prop)
{
    int n = rowCount();
    if (m_items.find(sockName) != m_items.end())
        return;
    insertRow(n, sockName, type, deflValue, prop);
}

void IParamModel::setItem(const QModelIndex& idx, const QString& type, const QVariant& deflValue, const PARAM_LINKS& links)
{
    setData(idx, type, ROLE_PARAM_TYPE);
    setData(idx, deflValue, ROLE_PARAM_VALUE);
    setData(idx, QVariant::fromValue(links), ROLE_PARAM_LINKS);
}

bool IParamModel::removeLink(const QString& sockName, const QModelIndex& linkIdx)
{
    QModelIndex idx = index(sockName);
    if (!idx.isValid())
        return false;

    QString name = nameFromRow(idx.row());
    auto itItem = m_items.find(name);
    ZASSERT_EXIT(itItem != m_items.end(), false);
    _ItemInfo& item = m_items[name];

    QVariant oldValue = QVariant::fromValue(item.links);

    item.links.removeOne(linkIdx);

    emit dataChanged(idx, idx, QVector<int>{ROLE_PARAM_LINKS});
}

QStringList IParamModel::sockNames() const
{
    QStringList names;
    for (int r = 0; r < rowCount(); r++)
    {
        ZASSERT_EXIT(m_row2Key.find(r) != m_row2Key.end(), names);
        names.append(m_row2Key[r]);
    }
    return names;
}

bool IParamModel::_insertRow(
    int row,
    const QString& sockName,
    const QString& type,
    const QVariant& deflValue,
    int prop)
{
    ZASSERT_EXIT(m_items.find(sockName) == m_items.end(), false);
    int nRows = m_items.size();

    _ItemInfo item;
    item.name = sockName;
    item.pConst = deflValue;
    item.type = type;
    item.prop = prop;

    const QString& nodeCls = m_nodeIdx.data(ROLE_OBJNAME).toString();
    NODE_DESC desc;
    m_model->getDescriptor(nodeCls, desc);

    if (type == "dict" || type == "DictObject" || type == "DictObject:NumericObject")
    {
        item.type = "dict";     //pay attention not to export to outside, only as a ui keyword.
        if (!desc.categories.contains("dict"))
            item.prop = SOCKPROP_DICTLIST_PANEL;
    }
    else if (type == "list")
    {
        if (!desc.categories.contains("list"))
            item.prop = SOCKPROP_DICTLIST_PANEL;
    } 
    else if (type == "group-line") 
    {
        if (!desc.categories.contains("group-line"))
            item.prop = SOCKPROP_GROUP_LINE;
    }

    //not type desc on list output socket, add it here.
    if (sockName == "list" && type.isEmpty())
    {
        item.type = "list";
        if (m_class == PARAM_INPUT && !desc.categories.contains("list"))
            item.prop = SOCKPROP_DICTLIST_PANEL;
    }

    //item.links = links;   //there will be not link info in INPUT_SOCKETS/OUTPUT_SOCKETS for safety.
    //and we will import the links by method GraphsModel::addLink.

    if (row == nRows)
    {
        //append
        m_items[sockName] = item;
        m_row2Key[nRows] = sockName;
        m_key2Row[sockName] = nRows;
    }
    else if (row < nRows)
    {
        auto itRow = m_row2Key.find(row);
        ZASSERT_EXIT(itRow != m_row2Key.end(), false);
        int nRows = rowCount();
        for (int r = nRows; r > row; r--)
        {
            const QString& key = m_row2Key[r - 1];
            m_row2Key[r] = key;
            m_key2Row[key] = r;
        }
        m_items[sockName] = item;
        m_row2Key[row] = sockName;
        m_key2Row[sockName] = row;
    }
    else
    {
        ZASSERT_EXIT(false, false);
    }

    // init dict key model.
    if ((item.prop & SOCKPROP_DICTLIST_PANEL) && m_items.find(sockName) != m_items.end())
    {
        DictKeyModel* pTblModel = new DictKeyModel(m_model, index(sockName), this);
        m_items[sockName].customData[ROLE_VPARAM_LINK_MODEL] = QVariantPtr<DictKeyModel>::asVariant(pTblModel);
        //connect(pTblModel, &DictKeyModel::rowsAboutToBeRemoved, this, &IParamModel::onKeyItemAboutToBeRemoved);
    }

    m_model->markDirty();
    return true;
}

void IParamModel::onKeyItemAboutToBeRemoved(const QModelIndex& parent, int first, int last)
{
    DictKeyModel* pTblModel = qobject_cast<DictKeyModel*>(sender());
    ZASSERT_EXIT(pTblModel);

    const QString& keyName = pTblModel->index(first, 0).data().toString();

    const QModelIndex& idxObj = pTblModel->index(first, 1);
    const QString& objId = idxObj.data().toString();

    m_bRetryLinkOp = true;
    zeno::scope_exit sp([this](){ m_bRetryLinkOp = false; });

    PARAM_LINKS links = idxObj.data(ROLE_PARAM_LINKS).value<PARAM_LINKS>();
    for (auto linkIdx : links)
    {
        m_model->removeLink(linkIdx, true);
    }
}
