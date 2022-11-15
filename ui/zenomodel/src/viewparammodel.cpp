#include "viewparammodel.h"
#include "parammodel.h"
#include "zassert.h"
#include "modelrole.h"


ProxySlotObject::ProxySlotObject(VParamItem* pItem, QObject* parent)
    : m_pItem(pItem)
    , QObject(parent)
{
}

ProxySlotObject::~ProxySlotObject()
{
    unmap();
}

void ProxySlotObject::onDataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles)
{
    if (topLeft == m_pItem->m_index) {
        QModelIndex viewIdx = m_pItem->index();
        emit m_pItem->model()->dataChanged(viewIdx, viewIdx, roles);
    }
}

void ProxySlotObject::unmap()
{
    if (m_pItem->m_index.isValid())
    {
        QObject::disconnect(m_pItem->m_index.model(), &QAbstractItemModel::dataChanged, this, &ProxySlotObject::onDataChanged);
    }
}

void ProxySlotObject::mapCoreIndex(const QPersistentModelIndex& idx)
{
    unmap();
    if (idx.isValid())
    {
        bool ret = QObject::connect(idx.model(), &QAbstractItemModel::dataChanged, this, &ProxySlotObject::onDataChanged);
        ZASSERT_EXIT(ret);
    }
}



VParamItem::VParamItem(VPARAM_TYPE type, const QString& text, bool bMapCore)
    : QStandardItem(text)
    , vType(type)
    , m_proxySlot(this)
{
    m_info.control = CONTROL_NONE;
    m_info.name = text;
}

VParamItem::VParamItem(VPARAM_TYPE type, const QIcon& icon, const QString& text, bool bMapCore)
    : QStandardItem(icon, text)
    , vType(type)
    , m_proxySlot(this)
{
    m_info.control = CONTROL_NONE;
    m_info.name = text;
}

VParamItem::VParamItem(const VParamItem& other)
    : QStandardItem(other)
    , vType(other.vType)
    , m_info(other.m_info)
    , m_proxySlot(this)
{
    mapCoreParam(other.m_index);
}

VParamItem::~VParamItem()
{
}

QVariant VParamItem::data(int role) const
{
    switch (role)
    {
    case Qt::EditRole:  return m_info.name;
    case Qt::DisplayRole:
    case ROLE_VPARAM_NAME:  return m_info.name;
    case ROLE_VPARAM_TYPE:  return vType;
    case ROLE_PARAM_CTRL:   return m_info.control;  //todo: remove control at core param.
    case ROLE_PARAM_NAME:
    {
        if (!m_index.isValid())
            return "";
        return m_index.data(ROLE_PARAM_NAME);
    }
    case ROLE_PARAM_VALUE:
    {
        if (!m_index.isValid())
            return m_info.value;
        return m_index.data(ROLE_PARAM_VALUE);
    }
    case ROLE_PARAM_TYPE:
    {
        if (!m_index.isValid())
            return m_info.typeDesc;
        return m_index.data(ROLE_PARAM_TYPE);
    }
    case ROLE_PARAM_LINKS:
    {
        if (!m_index.isValid())
            return QVariant();
        return m_index.data(ROLE_PARAM_LINKS);
    }
    case ROLE_VPARAM_IS_COREPARAM:
    {
        return m_index.isValid();
    }
    case ROLE_PARAM_SOCKETTYPE:
    {
        if (!m_index.isValid())
            return PARAM_UNKNOWN;
        return m_index.data(ROLE_PARAM_SOCKETTYPE);
    }
    case ROLE_OBJID:
    {
        return m_index.data(ROLE_OBJID);
    }
    case ROLE_VAPRAM_EDITTABLE:
    default:
        return QStandardItem::data(role);
    }
}

void VParamItem::setData(const QVariant& value, int role)
{
    switch (role)
    {
        case Qt::EditRole:
        case ROLE_VPARAM_NAME:
        {
            if (value == m_info.name)
                return;
            m_info.name = value.toString();
            break;
        }
        case ROLE_PARAM_NAME:
        {
            if (m_index.isValid())
            {
                QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(m_index.model());
                bool ret = pModel->setData(m_index, value, role);
                if (ret)
                {
                    //when core name updated, need to sync view name.
                    setData(value, ROLE_VPARAM_NAME);
                }
            }
            break;
        }
        case ROLE_PARAM_CTRL:
        {
            if (value == m_info.control)
                return;
            m_info.control = (PARAM_CONTROL)value.toInt();
            return;
        }
        case ROLE_PARAM_VALUE:
        {
            if (m_index.isValid())
            {
                QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(m_index.model());
                pModel->setData(m_index, value, role);
                return;
            }
            else
            {
                m_info.value = value;
            }
        }
        case ROLE_VAPRAM_EDITTABLE:
            break;
    }
    QStandardItem::setData(value, role);
}

VParamItem* VParamItem::getItem(const QString& uniqueName) const
{
    for (int r = 0; r < rowCount(); r++)
    {
        VParamItem* pChild = static_cast<VParamItem*>(child(r));
        if (pChild->m_info.name == uniqueName)
            return pChild;
    }
    return nullptr;
}

QStandardItem* VParamItem::clone() const
{
    VParamItem* pItem = new VParamItem(*this);
    return pItem;
}

void VParamItem::cloneChildren(VParamItem* rItem)
{
    if (!rItem)
        return;
    for (int r = 0; r < rItem->rowCount(); r++)
    {
        VParamItem* rChild = static_cast<VParamItem*>(rItem->child(r));
        VParamItem* newItem = new VParamItem(*rChild);
        newItem->cloneChildren(rChild);
        appendRow(newItem);
    }
}

void VParamItem::mapCoreParam(const QPersistentModelIndex& idx)
{
    m_index = idx;
    m_proxySlot.mapCoreIndex(idx);
}

bool VParamItem::operator==(VParamItem* rItem) const
{
    //only itself.
    if (!rItem) return false;
    return (rItem->m_info.name == m_info.name &&
            rItem->m_info.control == m_info.control &&
            rItem->vType == vType &&
            rItem->m_info.typeDesc == m_info.typeDesc &&
            rItem->m_index == m_index);
}



ViewParamModel::ViewParamModel(bool bNodeUI, QObject* parent)
    : QStandardItemModel(parent)
    , m_bNodeUI(bNodeUI)
{
    setup("");
}

ViewParamModel::ViewParamModel(bool bNodeUI, const QString& customXml, QObject* parent)
    : QStandardItemModel(parent)
    , m_bNodeUI(bNodeUI)
{
    setup(customXml);
}

void ViewParamModel::setup(const QString& customUI)
{
    if (customUI.isEmpty())
    {
        if (m_bNodeUI)
            initNode();
        else
            initPanel();
    }
}

void ViewParamModel::initPanel()
{
    /*default structure:
                root
                    |-- Tab (Default)
                        |-- Inputs (Group)
                            -- input param1 (Item)
                            -- input param2
                            ...

                        |-- Params (Group)
                            -- param1 (Item)
                            -- param2 (Item)
                            ...

                        |- Outputs (Group)
                            - output param1 (Item)
                            - output param2 (Item)
                ...
            */
    VParamItem* pRoot = new VParamItem(VPARAM_ROOT, "root");
    pRoot->setEditable(false);

    VParamItem* pTab = new VParamItem(VPARAM_TAB, "Default");
    {
        VParamItem* pInputsGroup = new VParamItem(VPARAM_GROUP, "In Sockets");
        VParamItem* paramsGroup = new VParamItem(VPARAM_GROUP, "Parameters");
        VParamItem* pOutputsGroup = new VParamItem(VPARAM_GROUP, "Out Sockets");

        pInputsGroup->setData(!m_bNodeUI, ROLE_VAPRAM_EDITTABLE);
        paramsGroup->setData(!m_bNodeUI, ROLE_VAPRAM_EDITTABLE);
        pOutputsGroup->setData(!m_bNodeUI, ROLE_VAPRAM_EDITTABLE);

        pTab->appendRow(pInputsGroup);
        pTab->appendRow(paramsGroup);
        pTab->appendRow(pOutputsGroup);
    }
    pTab->setData(!m_bNodeUI, ROLE_VAPRAM_EDITTABLE);

    pRoot->appendRow(pTab);
    appendRow(pRoot);
}

void ViewParamModel::initNode()
{
    /*default structure:
    |-- Inputs (Group)
        -- input param1 (Item)
        -- input param2
        ...

    |-- Params (Group)
        -- param1 (Item)
        -- param2 (Item)
        ...

    |- Outputs (Group)
        - output param1 (Item)
        - output param2 (Item)
        ...
    */
    VParamItem* pInputsGroup = new VParamItem(VPARAM_GROUP, "In Sockets");
    VParamItem* paramsGroup = new VParamItem(VPARAM_GROUP, "Parameters");
    VParamItem* pOutputsGroup = new VParamItem(VPARAM_GROUP, "Out Sockets");

    pInputsGroup->setData(!m_bNodeUI, ROLE_VAPRAM_EDITTABLE);
    paramsGroup->setData(!m_bNodeUI, ROLE_VAPRAM_EDITTABLE);
    pOutputsGroup->setData(!m_bNodeUI, ROLE_VAPRAM_EDITTABLE);

    appendRow(pInputsGroup);
    appendRow(paramsGroup);
    appendRow(pOutputsGroup);
}


QString ViewParamModel::exportUI() const
{
    /*
     xml example:

     <customui>
        <node name = "VDBWrangle">
            <tab name = "Default" type="default" hint="default msg for node">
                <group name = "inputs">
                    <param name = "..." control = "..."/>
                    ...
                </group>
                <group name = "params">
                    <param name = "..." control = "..."/>
                </group>
                <group name = "outputs">
                    
                </group>
            </tab>
        </node>
        <node name = ...>
        </node>
     </customui>
    */
    return QString();
}

void ViewParamModel::onParamsInserted(const QModelIndex& parent, int first, int last)
{
    IParamModel* pModel = qobject_cast<IParamModel*>(sender());
    ZASSERT_EXIT(pModel);
    const QModelIndex& idx = pModel->index(first, 0, parent);
    if (!idx.isValid()) return;

    QStandardItem* pRoot = invisibleRootItem();
    PARAM_CLASS cls = pModel->paramClass();
    if (cls == PARAM_INPUT)
    {
        QList<QStandardItem*> lst = findItems("In Sockets", Qt::MatchRecursive | Qt::MatchExactly);
        for (QStandardItem* pItem : lst)
        {
            if (pItem->data(ROLE_VPARAM_TYPE) == VPARAM_GROUP)
            {
                const QString& realName = idx.data(ROLE_PARAM_NAME).toString();
                const QString& displayName = realName;  //todo: mapping name.
                PARAM_CONTROL ctrl = (PARAM_CONTROL)idx.data(ROLE_PARAM_CTRL).toInt();
                VParamItem* paramItem = new VParamItem(VPARAM_PARAM, displayName, true);
                paramItem->m_info.control = ctrl;
                paramItem->mapCoreParam(idx);
                paramItem->setData(true, ROLE_VAPRAM_EDITTABLE);
                pItem->appendRow(paramItem);
                break;
            }
        }
    }
    else if (cls == PARAM_PARAM)
    {
        QList<QStandardItem*> lst = findItems("Parameters", Qt::MatchRecursive | Qt::MatchExactly);
        for (QStandardItem* pItem : lst)
        {
            if (pItem->data(ROLE_VPARAM_TYPE) == VPARAM_GROUP)
            {
                const QString& realName = idx.data(ROLE_PARAM_NAME).toString();
                const QString& displayName = realName;  //todo: mapping name.
                PARAM_CONTROL ctrl = (PARAM_CONTROL)idx.data(ROLE_PARAM_CTRL).toInt();
                VParamItem* paramItem = new VParamItem(VPARAM_PARAM, displayName, true);
                paramItem->m_info.control = ctrl;
                paramItem->mapCoreParam(idx);
                paramItem->setData(true, ROLE_VAPRAM_EDITTABLE);
                pItem->appendRow(paramItem);
                break;
            }
        }
    }
    else if (cls == PARAM_OUTPUT)
    {
        QList<QStandardItem*> lst = findItems("Out Sockets", Qt::MatchRecursive | Qt::MatchExactly);
        for (QStandardItem* pItem : lst)
        {
            if (pItem->data(ROLE_VPARAM_TYPE) == VPARAM_GROUP)
            {
                const QString& realName = idx.data(ROLE_PARAM_NAME).toString();
                const QString& displayName = realName;  //todo: mapping name.
                PARAM_CONTROL ctrl = (PARAM_CONTROL)idx.data(ROLE_PARAM_CTRL).toInt();
                VParamItem* paramItem = new VParamItem(VPARAM_PARAM, displayName, true);
                paramItem->m_info.control = ctrl;
                paramItem->mapCoreParam(idx);
                paramItem->setData(true, ROLE_VAPRAM_EDITTABLE);
                pItem->appendRow(paramItem);
                break;
            }
        }
    }
}

void ViewParamModel::onParamsAboutToBeRemoved(const QModelIndex& parent, int first, int last)
{
    IParamModel* pModel = qobject_cast<IParamModel*>(sender());
    ZASSERT_EXIT(pModel);
    const QModelIndex& idx = pModel->index(first, 0, parent);
    if (!idx.isValid()) return;
}

void ViewParamModel::onDataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles)
{
    //we use ProxySlotObject as a proxy to receive dataChanged from IParamModel.
}

void ViewParamModel::clone(ViewParamModel* pModel)
{
    if (m_bNodeUI)
    {
        QStandardItem* pRoot = invisibleRootItem();
        ZASSERT_EXIT(pRoot);
        pRoot->removeRows(0, 3);

        QStandardItem* pRightRoot = pModel->invisibleRootItem();
        for (int r = 0; r < pRightRoot->rowCount(); r++)
        {
            VParamItem* pRight = static_cast<VParamItem*>(pRightRoot->child(r));
            VParamItem* newItem = new VParamItem(*pRight);
            newItem->cloneChildren(pRight);
            pRoot->appendRow(newItem);
        }
    }
    else
    {
        QStandardItem* pRoot = invisibleRootItem();
        ZASSERT_EXIT(pRoot);

        pRoot->removeRow(0);

        VParamItem* pRight = static_cast<VParamItem*>(pModel->invisibleRootItem()->child(0));
        VParamItem* newItem = new VParamItem(*pRight);
        newItem->cloneChildren(pRight);

        pRoot->appendRow(newItem);
    }
}
