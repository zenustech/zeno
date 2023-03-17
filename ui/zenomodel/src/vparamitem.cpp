#include "vparamitem.h"
#include "viewparammodel.h"
#include "modelrole.h"
#include "uihelper.h"
#include "../customui/customuirw.h"
#include "nodeparammodel.h"
#include "iotags.h"


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
    if (topLeft == m_pItem->m_index && !roles.isEmpty())
    {
        int role = roles[0];
        QModelIndex viewIdx = m_pItem->index();
        if (role == ROLE_PARAM_TYPE)
        {
            const QString &newType = topLeft.data(ROLE_PARAM_TYPE).toString();
            PARAM_CONTROL newCtrl = UiHelper::getControlByType(newType);
            m_pItem->setData(newCtrl, ROLE_PARAM_CTRL);
            m_pItem->setData(newType, ROLE_PARAM_TYPE);
            emit m_pItem->model()->dataChanged(viewIdx, viewIdx, {ROLE_PARAM_CTRL});
            emit m_pItem->model()->dataChanged(viewIdx, viewIdx, roles);
        }
        else if (ROLE_PARAM_NAME == role)
        {
            m_pItem->setData(topLeft.data(ROLE_PARAM_NAME), ROLE_PARAM_NAME);
            emit m_pItem->model()->dataChanged(viewIdx, viewIdx, {ROLE_VPARAM_NAME});
            emit m_pItem->model()->dataChanged(viewIdx, viewIdx, roles);
        }
        else if (ROLE_PARAM_VALUE == role)
        {
            m_pItem->setData(topLeft.data(ROLE_PARAM_VALUE), ROLE_PARAM_VALUE);
            emit m_pItem->model()->dataChanged(viewIdx, viewIdx, roles);
        } 
		else if (ROLE_VPARAM_CTRL_PROPERTIES == role) 
		{
            m_pItem->m_customData[ROLE_VPARAM_CTRL_PROPERTIES] = topLeft.data(ROLE_VPARAM_CTRL_PROPERTIES);
            emit m_pItem->model()->dataChanged(viewIdx, viewIdx, roles);
        }
        else if (ROLE_VPARAM_TOOLTIP == role) 
        {
            m_pItem->m_customData[ROLE_VPARAM_TOOLTIP] = topLeft.data(ROLE_VPARAM_TOOLTIP);
            emit m_pItem->model()->dataChanged(viewIdx, viewIdx, roles);
        }
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
    m_ctrl = CONTROL_NONE;
    m_name = text;
    m_uuid = UiHelper::generateUuidInt();
}

VParamItem::VParamItem(VPARAM_TYPE type, const QIcon& icon, const QString& text, bool bMapCore)
    : QStandardItem(icon, text)
    , vType(type)
    , m_proxySlot(this)
{
    m_ctrl = CONTROL_NONE;
    m_name = text;
    m_uuid = UiHelper::generateUuidInt();
}

VParamItem::VParamItem(const VParamItem& other)
    : QStandardItem(other)
    , vType(other.vType)
    , m_name(other.m_name)
    , m_type(other.m_type)
    , m_value(other.m_value)
    , m_ctrl(other.m_ctrl)
    , m_sockProp(other.m_sockProp)
    /*, m_links(other.m_links)   link cannot be clone directly.*/
    , m_uuid(other.m_uuid)
    , m_proxySlot(this)
	, m_customData(other.m_customData)
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
    case Qt::DisplayRole:// return QStandardItem::data(role);
    case Qt::EditRole:
    case ROLE_VPARAM_NAME:  return m_name;
    case ROLE_VPARAM_TYPE:  return vType;
    case ROLE_PARAM_CTRL:   return m_ctrl;
    case ROLE_PARAM_NAME:
    {
        if (!m_index.isValid())
            return m_name;
        return m_index.data(ROLE_PARAM_NAME);
    }
    case ROLE_PARAM_VALUE:
    {
        if (!m_index.isValid())
            return m_value;
        return m_index.data(ROLE_PARAM_VALUE);
    }
    case ROLE_PARAM_TYPE:
    {
        if (!m_index.isValid())
            return m_type;
        return m_index.data(ROLE_PARAM_TYPE);
    }
    case ROLE_PARAM_SOCKPROP:
    {
        if (!m_index.isValid())
            return m_sockProp;
        return m_index.data(ROLE_PARAM_SOCKPROP);
    }
    case ROLE_PARAM_COREIDX:
    {
        return m_index;
    }
    case ROLE_VPARAM_LINK_MODEL:
    {
        if (!m_index.isValid())
        {
            if (m_customData.find(role) != m_customData.end())
            {
                return m_customData[role];
            }
            break;
        }
        return m_index.data(ROLE_VPARAM_LINK_MODEL);
    }
    case ROLE_PARAM_LINKS:
    {
        if (!m_index.isValid())
            return QVariant::fromValue(m_links);
        return m_index.data(ROLE_PARAM_LINKS);
    }
    case ROLE_VPARAM_IS_COREPARAM:
    {
        return m_index.isValid();
    }
    case ROLE_PARAM_CLASS:
    {
        if (!m_index.isValid())
            return PARAM_UNKNOWN;
        return m_index.data(ROLE_PARAM_CLASS);
    }
    case ROLE_OBJID:
    case ROLE_OBJPATH:
    {
        if (model())
            return model()->data(index(), role);
        return "";
    }
    case ROLE_ADDLINK:
    case ROLE_REMOVELINK:
        break;  //todo: link for custom param.
    case ROLE_VPARAM_CTRL_PROPERTIES:
        if (m_customData.find(role) != m_customData.end())
        {
            return m_customData[role];
        } 
    case ROLE_VAPRAM_EDITTABLE: 
    {
        if (m_customData.find(role) != m_customData.end()) {
            return m_customData[role];
        } else {
            return true;
        }
    }
    case ROLE_VPARAM_TOOLTIP: 
    {
        if (m_customData.find(role) != m_customData.end()) 
        {
            return m_customData[role];
        } else 
        {
            return "";
        }
    }
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
            if (value == m_name)
                return;

            const QModelIndex& idx = index();
            ViewParamModel* pModel = qobject_cast<ViewParamModel *>(model());
            QString oldPath = model()->data(idx, ROLE_OBJPATH).toString();

            m_name = value.toString();

            QString newPath = model()->data(idx, ROLE_OBJPATH).toString();
            if (role == Qt::EditRole && oldPath != newPath && pModel)
            {
                emit pModel->editNameChanged(idx, oldPath, m_name);
            }
            if (pModel && !m_index.isValid())
                pModel->markDirty();
            break;
        }
        case ROLE_PARAM_NAME:
        {
            m_name = value.toString();
            break;
        }
        case ROLE_PARAM_CTRL:
        {
            if (value == m_ctrl)
                return;
            m_ctrl = (PARAM_CONTROL)value.toInt();
            auto viewModel = qobject_cast<ViewParamModel*>(model());
            if (viewModel && !m_index.isValid())
                viewModel->markDirty();
            break;
        }
        case ROLE_PARAM_TYPE:
        {
            if (m_type == value.toString())
                return;
            m_type = value.toString();
            break;
        }
        case ROLE_PARAM_VALUE:
        {
            if (value == m_value)
                return;

            if (m_index.isValid())
            {
                QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(m_index.model());
                pModel->setData(m_index, value, role);
            }
            m_value = value;
            auto viewModel = qobject_cast<ViewParamModel*>(model());
            if (viewModel && !m_index.isValid())
                viewModel->markDirty();
            break;
        }
        case ROLE_PARAM_COREIDX:
        {
            QPersistentModelIndex coreIdx = value.toPersistentModelIndex();
            if (coreIdx == m_index)
                return;
            mapCoreParam(coreIdx);
            break;
        }
        case ROLE_ADDLINK:
        case ROLE_REMOVELINK:
        {
            if (vType != VPARAM_PARAM)
                return;

            if (m_index.isValid())
            {
                QAbstractItemModel *pModel = const_cast<QAbstractItemModel *>(m_index.model());
                bool ret = pModel->setData(m_index, value, role);
            }
            QPersistentModelIndex linkIdx = value.toPersistentModelIndex();
            if (role == ROLE_ADDLINK) {
                m_links.append(linkIdx);
            }
            else {
                m_links.removeAll(linkIdx);
            }
            break;
        }
        case ROLE_VPARAM_CTRL_PROPERTIES:
        case ROLE_VAPRAM_EDITTABLE: 
        case ROLE_VPARAM_TOOLTIP: 
        {
            m_customData[role] = value;
            break;
        }
    }
    QStandardItem::setData(value, role);
}

void VParamItem::read(QDataStream& in)
{
    QString paramName;
    in >> paramName;

    QString jsonParam;
    in >> jsonParam;

    rapidjson::Document doc;
    QByteArray bytes = jsonParam.toUtf8();
    doc.Parse(bytes);

    VPARAM_INFO param = zenomodel::importParam(paramName, doc.GetObject());
    m_tempInfo = param;
}

void VParamItem::write(QDataStream& out) const
{
    rapidjson::StringBuffer s;
    RAPIDJSON_WRITER writer(s);
    zenomodel::exportItem(this, writer);

    QString content = QString(s.GetString());
    out << content;
}

VParamItem* VParamItem::getItem(const QString& uniqueName, int* targetIdx) const
{
    for (int r = 0; r < rowCount(); r++)
    {
        VParamItem* pChild = static_cast<VParamItem*>(child(r));
        if (pChild->m_name == uniqueName)
        {
            if (targetIdx)
                *targetIdx = r;
            return pChild;
        }
    }
    return nullptr;
}

VParamItem* VParamItem::findItem(uint uuid, int* targetIdx) const
{
    for (int r = 0; r < rowCount(); r++)
    {
        VParamItem* pChild = static_cast<VParamItem*>(child(r));
        if (pChild->m_uuid == uuid)
        {
            if (targetIdx)
                *targetIdx = r;
            return pChild;
        }
        else if (VParamItem* pTarget = pChild->findItem(uuid, targetIdx))
        {
            return pTarget;
        }
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

void VParamItem::cloneFrom(VParamItem* pOther)
{
    if (!pOther)
        return;

    m_index = pOther->m_index;
    m_name = pOther->m_name;
    m_type = pOther->m_type;
    m_ctrl = pOther->m_ctrl;
    m_value = pOther->m_value;

    vType = pOther->vType;
    m_tempInfo = pOther->m_tempInfo;
    m_links = pOther->m_links;
    m_sockProp = pOther->m_sockProp;

    m_uuid = pOther->m_uuid;
    m_customData = pOther->m_customData;

    setData(m_name, Qt::DisplayRole);
    //todo: data
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
    return (rItem->m_name == m_name &&
            rItem->m_ctrl == m_ctrl &&
            rItem->vType == vType &&
            rItem->m_type == m_type &&
            rItem->m_index == m_index);
}

void VParamItem::importParamInfo(const VPARAM_INFO& paramInfo)
{
    if (paramInfo.vType == VPARAM_ROOT)
    {
        this->vType = paramInfo.vType;
        //this->m_info = paramInfo.m_info;
        for (VPARAM_INFO tab : paramInfo.children)
        {

        }
    }
}

PARAM_CLASS VParamItem::getParamClass()
{
    if (vType != VPARAM_PARAM)
        return PARAM_UNKNOWN;

    NodeParamModel* pModel = qobject_cast<NodeParamModel*>(this->model());
    if (!pModel)
        return PARAM_UNKNOWN;

    VParamItem* parentItem = static_cast<VParamItem*>(this->parent());
    ZASSERT_EXIT(parentItem, PARAM_UNKNOWN);

    if (parentItem->m_name == iotags::params::node_inputs)
        return PARAM_INPUT;
    else if (parentItem->m_name == iotags::params::node_params)
        return PARAM_PARAM;
    else if (parentItem->m_name == iotags::params::node_outputs)
        return PARAM_OUTPUT;
    else
        return PARAM_UNKNOWN;
}

VPARAM_INFO VParamItem::exportParamInfo()
{
    VPARAM_INFO info;
    if (vType == VPARAM_ROOT)
    {
        VPARAM_INFO root;
        root.m_cls = PARAM_UNKNOWN;
        root.vType = VPARAM_ROOT;
        for (int i = 0; i < rowCount(); i++)
        {
            VParamItem* pTab = static_cast<VParamItem *>(child(i));
            VPARAM_INFO tab = pTab->exportParamInfo();
            root.children.append(tab);
        }
        return root;
    }
    else if (vType == VPARAM_TAB)
    {
        VPARAM_INFO tab;
        tab.m_cls = PARAM_UNKNOWN;
        tab.vType = VPARAM_TAB;
        tab.m_info.name = data(ROLE_VPARAM_NAME).toString();
        for (int j = 0; j < rowCount(); j++)
        {
            VParamItem* pGroup =static_cast<VParamItem*>(child(j));
            VPARAM_INFO group = pGroup->exportParamInfo();
            tab.children.append(group);
        }
        return tab;
    }
    else if (vType == VPARAM_GROUP)
    {
        VPARAM_INFO group;
        group.m_cls = PARAM_UNKNOWN;
        group.vType = VPARAM_GROUP;
        group.m_info.name = data(ROLE_VPARAM_NAME).toString();

        for (int k = 0; k < rowCount(); k++)
        {
            VParamItem* pParam = static_cast<VParamItem *>(child(k));
            VPARAM_INFO param = pParam->exportParamInfo();
            group.children.append(param);
        }
        return group;
    }
    else if (vType == VPARAM_PARAM)
    {
        VPARAM_INFO param;

        VParamItem* parentItem = static_cast<VParamItem*>(this->parent());
        ZASSERT_EXIT(parentItem, info);
        const QString& groupName = parentItem->m_name;

        if (groupName == "In Sockets")
        {
            param.m_cls = PARAM_INPUT;
        }
        else if (groupName == "Parameters")
        {
            param.m_cls = PARAM_PARAM;
        }
        else if (groupName == "Out Sockets")
        {
            param.m_cls = PARAM_OUTPUT;
        }
        else
        {
            param.m_cls = PARAM_UNKNOWN;
        }
        param.vType = VPARAM_PARAM;
        param.controlInfos = data(ROLE_VPARAM_CTRL_PROPERTIES);
        QString refPath;
        if (m_index.isValid())
            refPath = m_index.data(ROLE_OBJPATH).toString();
        param.refParamPath = refPath;
        param.m_info.control = m_ctrl;
        param.m_info.defaultValue = m_value;
        param.m_info.name = m_name;
        param.m_info.typeDesc = m_type;
        return param;
    }
    else
    {
        ZASSERT_EXIT(false, info);
        return info;
    }
}