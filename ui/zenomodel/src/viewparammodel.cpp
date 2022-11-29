#include "viewparammodel.h"
#include "parammodel.h"
#include "zassert.h"
#include "modelrole.h"
#include <zenomodel/include/uihelper.h>
#include "variantptr.h"


static const char* qsToString(const QString& qs)
{
    std::string s = qs.toStdString();
    char* wtf = new char[s.size() + 1];
    strcpy(wtf, s.c_str());
    return wtf;
}


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
    if (topLeft == m_pItem->m_index)
    {
        QModelIndex viewIdx = m_pItem->index();
        if (roles.contains(ROLE_PARAM_TYPE))
        {
            const QString& newType = topLeft.data(ROLE_PARAM_TYPE).toString();
            PARAM_CONTROL newCtrl = UiHelper::getControlType(newType);
            m_pItem->m_info.control = newCtrl;
            emit m_pItem->model()->dataChanged(viewIdx, viewIdx, {ROLE_PARAM_CTRL});
        }
        if (roles.contains(ROLE_PARAM_NAME))
        {
            m_pItem->m_info.name = topLeft.data(ROLE_PARAM_NAME).toString();
            emit m_pItem->model()->dataChanged(viewIdx, viewIdx, { ROLE_VPARAM_NAME });
        }
        if (roles.contains(ROLE_PARAM_CTRL))
        {
            PARAM_CONTROL ctrl = (PARAM_CONTROL)topLeft.data(ROLE_PARAM_CTRL).toInt();
            m_pItem->m_info.control = ctrl;
        }
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
        if (model())
            return model()->data(index(), ROLE_OBJID);
        return "";
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
            qobject_cast<ViewParamModel*>(model())->markDirty();
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
            if (m_index.isValid())
            {
                QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(m_index.model());
                pModel->setData(m_index, value, role);
            }
            qobject_cast<ViewParamModel*>(model())->markDirty();
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
                qobject_cast<ViewParamModel*>(model())->markDirty();
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
    for (int r = 0; r < rowCount(); r++)
    {
        VParamItem* pChild = static_cast<VParamItem*>(child(r));
        QStandardItem* newItem = pChild->clone();
        pItem->appendRow(newItem);
    }
    return pItem;
}

void VParamItem::mapCoreParam(const QPersistentModelIndex& idx)
{
    m_index = idx;
    m_proxySlot.mapCoreIndex(idx);
}

rapidxml::xml_node<>* VParamItem::exportXml(rapidxml::xml_document<>& doc)
{
    int vType = data(ROLE_VPARAM_TYPE).toInt();
    if (vType == VPARAM_TAB)
    {
        const QString& name = data(ROLE_VPARAM_NAME).toString();
        rapidxml::xml_node<>* node = doc.allocate_node(rapidxml::node_element, "tab", qsToString(name));
        for (int r = 0; r < rowCount(); r++)
        {
            VParamItem* pChild = static_cast<VParamItem*>(child(r));
            ZASSERT_EXIT(pChild && pChild->data(ROLE_VPARAM_TYPE) == VPARAM_GROUP, nullptr);
            auto pNode = pChild->exportXml(doc);
            node->append_node(pNode);
        }
        return node;
    }
    else if (vType == VPARAM_GROUP)
    {
        const QString& name = data(ROLE_VPARAM_NAME).toString();
        rapidxml::xml_node<>* node = doc.allocate_node(rapidxml::node_element, "group", qsToString(name));
        for (int r = 0; r < rowCount(); r++)
        {
            VParamItem* pChild = static_cast<VParamItem*>(child(r));
            ZASSERT_EXIT(pChild && pChild->data(ROLE_VPARAM_TYPE) == VPARAM_PARAM, nullptr);
            auto pNode = pChild->exportXml(doc);
            node->append_node(pNode);
        }
        return node;
    }
    else if (vType == VPARAM_PARAM)
    {
        const QString& name = data(ROLE_VPARAM_NAME).toString();
        const QString& corename = data(ROLE_PARAM_NAME).toString();
        rapidxml::xml_node<>* node = doc.allocate_node(rapidxml::node_element, "param");
        //attributes
        node->append_attribute(doc.allocate_attribute("name", qsToString(name)));
        node->append_attribute(doc.allocate_attribute("coremap", qsToString(corename)));
        //control
        
        {
            rapidxml::xml_node<>* ctrlNode = doc.allocate_node(rapidxml::node_element, "control");
            ctrlNode->append_attribute(doc.allocate_attribute("type", "enum"));
            PARAM_CONTROL ctrl = (PARAM_CONTROL)data(ROLE_PARAM_CTRL).toInt();
            switch (ctrl)
            {
                case CONTROL_INT:
                case CONTROL_FLOAT:
                {
                    QString qsValue = QString::number(data(ROLE_PARAM_VALUE).toFloat());
                    rapidxml::xml_node<>* valueNode = doc.allocate_node(rapidxml::node_element, "value", qsToString(qsValue));
                    ctrlNode->append_node(valueNode);
                    break;
                }
            }
            node->append_node(ctrlNode);
        }
        //todo: link

        return node;
    }
    else
    {
        return nullptr;
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



ViewParamModel::ViewParamModel(bool bNodeUI, const QModelIndex& nodeIdx, IGraphsModel* pModel, QObject* parent)
    : QStandardItemModel(parent)
    , m_bNodeUI(bNodeUI)
    , m_nodeIdx(nodeIdx)
    , m_model(pModel)
    , m_bDirty(false)
{
    setup("");
}

void ViewParamModel::setup(const QString& customUI)
{
    if (customUI.isEmpty())
    {
        initCustomUI();
    }
}

void ViewParamModel::initCustomUI()
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

QModelIndex ViewParamModel::indexFromPath(const QString& path)
{
    QStringList lst = path.split("/", Qt::SkipEmptyParts);
    if (lst.size() == 4)
    {
        const QString& root = lst[0];
        const QString& tab = lst[1];
        const QString& group = lst[2];
        const QString& param = lst[3];
        if (root != "root") return QModelIndex();

        QStandardItem* rootItem = invisibleRootItem()->child(0);
        if (!rootItem) return QModelIndex();

        for (int i = 0; i < rootItem->rowCount(); i++)
        {
            QStandardItem* pTab = rootItem->child(i);
            if (pTab->data(ROLE_VPARAM_NAME) == tab)
            {
                for (int j = 0; j < pTab->rowCount(); j++)
                {
                    QStandardItem* pGroup = pTab->child(j);
                    if (pGroup->data(ROLE_VPARAM_NAME) == group)
                    {
                        for (int k = 0; k < pGroup->rowCount(); k++)
                        {
                            QStandardItem* pParam = pGroup->child(k);
                            if (pParam->data(ROLE_VPARAM_NAME) == param)
                            {
                                return pParam->index();
                            }
                        }
                    }
                }
            }
        }
    }
    return QModelIndex();
}

void ViewParamModel::resetParams(const VPARAM_INFO& invisibleRoot)
{
    //clear old data
    this->clear();

    VParamItem* pRoot = new VParamItem(VPARAM_ROOT, "root");
    for (VPARAM_INFO tab : invisibleRoot.children)
    {
        VParamItem* pTabItem = new VParamItem(VPARAM_TAB, tab.m_info.name);
        for (VPARAM_INFO group : tab.children)
        {
            VParamItem* pGroupItem = new VParamItem(VPARAM_GROUP, group.m_info.name);
            for (VPARAM_INFO param : group.children)
            {
                const QString& paramName = param.m_info.name;
                VParamItem* paramItem = new VParamItem(VPARAM_PARAM, paramName);

                //mapping core.
                const QString& coreparam = param.coreParam;
                if (!coreparam.isEmpty())
                {
                    if (param.m_cls == PARAM_INPUT)
                    {
                        IParamModel* inputsModel = QVariantPtr<IParamModel>::asPtr(m_nodeIdx.data(ROLE_INPUT_MODEL));
                        paramItem->m_index = inputsModel->index(coreparam);
                    }
                    else if (param.m_cls == PARAM_PARAM)
                    {
                        IParamModel* paramsModel = QVariantPtr<IParamModel>::asPtr(m_nodeIdx.data(ROLE_PARAM_MODEL));
                        paramItem->m_index = paramsModel->index(coreparam);
                    }
                    else if (param.m_cls == PARAM_OUTPUT)
                    {
                        IParamModel* outputsModel = QVariantPtr<IParamModel>::asPtr(m_nodeIdx.data(ROLE_OUTPUT_MODEL));
                        paramItem->m_index = outputsModel->index(coreparam);
                    }
                }
                paramItem->m_info = param.m_info;
                paramItem->setData(param.controlInfos, ROLE_VPARAM_CTRL_PROPERTIES);

                pGroupItem->appendRow(paramItem);
            }
            pTabItem->appendRow(pGroupItem);
        }
        pRoot->appendRow(pTabItem);
    }
    invisibleRootItem()->appendRow(pRoot);
    markDirty();
}

void ViewParamModel::onCoreParamsInserted(const QModelIndex& parent, int first, int last)
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

void ViewParamModel::onCoreParamsAboutToBeRemoved(const QModelIndex& parent, int first, int last)
{
    IParamModel* pModel = qobject_cast<IParamModel*>(sender());
    ZASSERT_EXIT(pModel);
    const QModelIndex& idx = pModel->index(first, 0, parent);
    if (!idx.isValid())
        return;

    QStandardItem* rootItem = invisibleRootItem()->child(0);
    if (!rootItem)
        return;

    for (int i = 0; i < rootItem->rowCount(); i++)
    {
        QStandardItem* pTab = rootItem->child(i);
        for (int j = 0; j < pTab->rowCount(); j++)
        {
            QStandardItem* pGroup = pTab->child(j);
            for (int k = 0; k < pGroup->rowCount(); k++)
            {
                VParamItem* pParam = static_cast<VParamItem*>(pGroup->child(k));
                if (pParam->m_index == idx)
                {
                    pGroup->removeRow(k);
                    return;
                }
            }
        }
    }
}

QPersistentModelIndex ViewParamModel::nodeIdx() const
{
    return m_nodeIdx;
}

QVariant ViewParamModel::data(const QModelIndex& index, int role) const
{
    switch (role)
    {
        case ROLE_OBJID:    return m_nodeIdx.isValid() ? m_nodeIdx.data(role) : "";
        case ROLE_OBJPATH:
        {
            QString path;
            QStandardItem* pItem = itemFromIndex(index);
            do
            {
                path = pItem->data(ROLE_VPARAM_NAME).toString() + path;
                path = "/" + path;
                pItem = pItem->parent();
            } while (pItem);
            if (m_bNodeUI) {
                path = QString("node-param") + cPathSeperator + path;
            }
            else {
                path = QString("panel-param") + cPathSeperator + path;
            }
            path = m_nodeIdx.data(ROLE_OBJPATH).toString() + cPathSeperator + path;
            return path;
        }
    }
    return QStandardItemModel::data(index, role);
}

bool ViewParamModel::isNodeModel() const
{
    return m_bNodeUI;
}

bool ViewParamModel::isDirty() const
{
    return m_bDirty;
}

void ViewParamModel::markDirty()
{
    m_bDirty = true;
}

void ViewParamModel::clone(ViewParamModel* pModel)
{
    QStandardItem* pRoot = invisibleRootItem();
    ZASSERT_EXIT(pRoot);
    pRoot->removeRows(0, pRoot->rowCount());

    QStandardItem* pRightRoot = pModel->invisibleRootItem();
    for (int r = 0; r < pRightRoot->rowCount(); r++)
    {
        QStandardItem* newItem = pRightRoot->child(r)->clone();
        pRoot->appendRow(newItem);
    }
}
