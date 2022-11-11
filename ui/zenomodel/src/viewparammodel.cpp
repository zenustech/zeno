#include "viewparammodel.h"
#include "parammodel.h"
#include "zassert.h"


VParamItem::VParamItem(VPARAM_TYPE type, const QString& text, bool bMapCore)
    : QStandardItem(text)
    , vType(type)
{
    m_info.control = CONTROL_NONE;
    m_info.name = text;
}

VParamItem::VParamItem(VPARAM_TYPE type, const QIcon& icon, const QString& text, bool bMapCore)
    : QStandardItem(icon, text)
    , vType(type)
{
    m_info.control = CONTROL_NONE;
    m_info.name = text;
}

VParamItem::VParamItem(const VParamItem& other)
    : QStandardItem(other)
    , vType(other.vType)
    , m_info(other.m_info)
    , m_index(other.m_index)
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
    case ROLE_PARAM_CTRL:   return m_info.control;
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
            return;
        }
        case ROLE_PARAM_VALUE:
        {
            if (m_index.isValid())
            {
                QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(m_index.model());
                pModel->setData(m_index, value, role);
            }
            else
            {
                m_info.value = value;
            }
            return;
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



ViewParamModel::ViewParamModel(QObject* parent)
    : QStandardItemModel(parent)
{
    setup("");
}

ViewParamModel::ViewParamModel(const QString& customXml, QObject* parent)
    : QStandardItemModel(parent)
{
    setup(customXml);
}

void ViewParamModel::setup(const QString& customUI)
{
    if (customUI.isEmpty())
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

        VParamItem* pTab = new VParamItem(VPARAM_TAB, "Default");
        {
            VParamItem* pInputsGroup = new VParamItem(VPARAM_GROUP, "In Sockets");
            VParamItem* paramsGroup = new VParamItem(VPARAM_GROUP, "Parameters");
            VParamItem* pOutputsGroup = new VParamItem(VPARAM_GROUP, "Out Sockets");

            pInputsGroup->setData(true, ROLE_VAPRAM_EDITTABLE);
            paramsGroup->setData(true, ROLE_VAPRAM_EDITTABLE);
            pOutputsGroup->setData(true, ROLE_VAPRAM_EDITTABLE);

            pTab->appendRow(pInputsGroup);
            pTab->appendRow(paramsGroup);
            pTab->appendRow(pOutputsGroup);
        }
        pTab->setData(true, ROLE_VAPRAM_EDITTABLE);

        pRoot->appendRow(pTab);
        pRoot->setData(true, ROLE_VAPRAM_EDITTABLE);

        appendRow(pRoot);
    }
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
                paramItem->m_index = idx;
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
                paramItem->m_index = idx;
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
                paramItem->m_index = idx;
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

void ViewParamModel::clone(ViewParamModel* pModel)
{
    QStandardItem* pRoot = invisibleRootItem();
    ZASSERT_EXIT(pRoot);

    pRoot->removeRow(0);

    VParamItem* pRight = static_cast<VParamItem*>(pModel->invisibleRootItem()->child(0));
    VParamItem* newItem = new VParamItem(*pRight);
    newItem->cloneChildren(pRight);

    pRoot->appendRow(newItem);
}
