#include "vparamitem.h"
#include "viewparammodel.h"
#include "modelrole.h"
#include "uihelper.h"
#include "../customui/customuirw.h"


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
            m_pItem->m_info.control = newCtrl;
            m_pItem->m_info.typeDesc = newType;
            emit m_pItem->model()->dataChanged(viewIdx, viewIdx, {ROLE_PARAM_CTRL});
            emit m_pItem->model()->dataChanged(viewIdx, viewIdx, roles);
        }
        else if (ROLE_PARAM_NAME == role)
        {
            m_pItem->m_info.name = topLeft.data(ROLE_PARAM_NAME).toString();
            emit m_pItem->model()->dataChanged(viewIdx, viewIdx, {ROLE_VPARAM_NAME});
            emit m_pItem->model()->dataChanged(viewIdx, viewIdx, roles);
        }
        else if (ROLE_PARAM_VALUE == role)
        {
            m_pItem->m_info.value = topLeft.data(ROLE_PARAM_VALUE);
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
            return m_info.name;
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
    case ROLE_PARAM_SOCKPROP:
    {
        if (!m_index.isValid())
            return SOCKPROP_UNKNOWN;
        return m_index.data(ROLE_PARAM_SOCKPROP);
    }
    case ROLE_PARAM_COREIDX:
    {
        return m_index;
    }
    case ROLE_VPARAM_LINK_MODEL:
    {
        if (!m_index.isValid())
            return QVariant();
        return m_index.data(ROLE_VPARAM_LINK_MODEL);
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

            const QModelIndex& idx = index();
            ViewParamModel* pModel = qobject_cast<ViewParamModel *>(model());
            QString oldPath = model()->data(idx, ROLE_OBJPATH).toString();

            m_info.name = value.toString();

            QString newPath = model()->data(idx, ROLE_OBJPATH).toString();
            if (role == Qt::EditRole && oldPath != newPath)
            {
                emit pModel->editNameChanged(idx, oldPath, m_info.name);
            }
            pModel->markDirty();
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
            qobject_cast<ViewParamModel*>(model())->markDirty();
            break;
        }
        case ROLE_PARAM_TYPE:
        {
            if (m_index.isValid())
            {
                //only update core type by editing SubInput/SubOutput.
            }
            m_info.typeDesc = value.toString();
            break;
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
                if (value == m_info.value)
                    return;
                m_info.value = value;
                qobject_cast<ViewParamModel*>(model())->markDirty();
            }
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
            if (m_index.isValid())
            {
                QAbstractItemModel *pModel = const_cast<QAbstractItemModel *>(m_index.model());
                bool ret = pModel->setData(m_index, value, role);
            }
            QModelIndex idx = index();
            emit this->model()->dataChanged(idx, idx, {role});
            break;
        }
        case ROLE_VAPRAM_EDITTABLE:
            break;
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
        if (pChild->m_info.name == uniqueName)
        {
            if (targetIdx)
                *targetIdx = r;
            return pChild;
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