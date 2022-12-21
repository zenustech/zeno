#include "viewparammodel.h"
#include "parammodel.h"
#include "zassert.h"
#include "modelrole.h"
#include <zenomodel/include/uihelper.h>
#include "variantptr.h"
#include <zenomodel/customui/customuirw.h>
#include "globalcontrolmgr.h"


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
    case ROLE_PARAM_SOCKPROP:
    {
        if (!m_index.isValid())
            return SOCKPROP_UNKNOWN;
        return m_index.data(ROLE_PARAM_SOCKPROP);
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
        case ROLE_VPARAM_COREIDX:
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

#ifdef ENABLE_DRAG_DROP_ITEM

QMimeData *ViewParamModel::mimeData(const QModelIndexList &indexes) const {
    QMimeData *mimeD = QAbstractItemModel::mimeData(indexes);
    if (indexes.size() > 0) {
        QModelIndex index = indexes.at(0);
        VParamItem *node = (VParamItem *)itemFromIndex(index);
        QByteArray encoded;
        QDataStream stream(&encoded, QIODevice::WriteOnly);
        stream << index.row() << index.column();
        stream << index.data(ROLE_VPARAM_NAME).toString();
        /*stream << node->index();*/
        //encodeData(indexes, stream);
        node->write(stream);
        const QString format = QStringLiteral("application/x-qstandarditemmodeldatalist");
        mimeD->setData(format, encoded);
    } else {
        mimeD->setData("Node/NodePtr", "NULL");
    }
    return mimeD;
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

Qt::ItemFlags ViewParamModel::flags(const QModelIndex& index) const
{
    Qt::ItemFlags itemFlags = QStandardItemModel::flags(index);
    VParamItem* item = static_cast<VParamItem*>(itemFromIndex(index));
    if (!item)
        return itemFlags;

    if (item->vType == VPARAM_PARAM)
    {
        itemFlags &= ~Qt::ItemIsDropEnabled;
    }
    else if (item->vType == VPARAM_GROUP)
    {
        itemFlags &= ~Qt::ItemIsDragEnabled;
    }
    else
    {
        itemFlags &= ~(Qt::ItemIsDragEnabled | Qt::ItemIsDropEnabled);
    }
    return itemFlags;
}

bool ViewParamModel::dropMimeData(const QMimeData* data, Qt::DropAction action, int row, int column, const QModelIndex& parent)
{
    bool ret = QStandardItemModel::dropMimeData(data, action, row, column, parent);
    if (!ret)
        return ret;

    QModelIndex newVParamIdx = this->index(row, column, parent);
    if (!newVParamIdx.isValid())
        return ret;

    VParamItem* pItem = static_cast<VParamItem*>(itemFromIndex(newVParamIdx));
    //mapping core.
    const QString& coreparam = pItem->m_tempInfo.coreParam;
    pItem->m_info = pItem->m_tempInfo.m_info;
    if (!coreparam.isEmpty())
    {
        if (pItem->m_tempInfo.m_cls == PARAM_INPUT)
        {
            IParamModel* inputsModel = QVariantPtr<IParamModel>::asPtr(this->data(QModelIndex(), ROLE_INPUT_MODEL));
            pItem->m_index = inputsModel->index(coreparam);
        }
        else if (pItem->m_tempInfo.m_cls == PARAM_PARAM)
        {
            IParamModel* paramsModel = QVariantPtr<IParamModel>::asPtr(this->data(QModelIndex(), ROLE_PARAM_MODEL));
            pItem->m_index = paramsModel->index(coreparam);
        }
        else if (pItem->m_tempInfo.m_cls == PARAM_OUTPUT)
        {
            IParamModel* outputsModel = QVariantPtr<IParamModel>::asPtr(this->data(QModelIndex(), ROLE_OUTPUT_MODEL));
            pItem->m_index = outputsModel->index(coreparam);
        }
        pItem->setData(true, ROLE_VPARAM_IS_COREPARAM);

        //copy inner type and value.
        if (pItem->m_index.isValid())
        {
            pItem->m_info.typeDesc = pItem->m_index.data(ROLE_PARAM_TYPE).toString();
            pItem->m_info.value = pItem->m_index.data(ROLE_PARAM_VALUE);
        }
    }
    else
    {
        pItem->setData(false, ROLE_VPARAM_IS_COREPARAM);
    }
    
    pItem->setData(pItem->m_tempInfo.controlInfos, ROLE_VPARAM_CTRL_PROPERTIES);
    return true;
}

void VParamItem::write(QDataStream& out) const
{
    rapidjson::StringBuffer s;
    RAPIDJSON_WRITER writer(s);
    zenomodel::exportItem(this, writer);

    QString content = QString(s.GetString());
    out << content;
}

#endif

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
    setItemPrototype(new VParamItem(VPARAM_PARAM, ""));
}

ViewParamModel::~ViewParamModel()
{
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

QModelIndex ViewParamModel::indexFromName(PARAM_CLASS cls, const QString &coreParam)
{
    QStandardItem* rootItem = invisibleRootItem()->child(0);
    if (!rootItem)
        return QModelIndex();

    for (int i = 0; i < rootItem->rowCount(); i++)
    {
        QStandardItem* pTab = rootItem->child(i);
        for (int j = 0; j < pTab->rowCount(); j++)
        {
            QStandardItem* pGroup = pTab->child(j);
            if (cls == PARAM_INPUT && pGroup->data(ROLE_VPARAM_NAME).toString() == "In Sockets")
            {
                for (int k = 0; k < pGroup->rowCount(); k++)
                {
                    VParamItem* pParam = static_cast<VParamItem*>(pGroup->child(k));
                    if (pParam->data(ROLE_PARAM_NAME) == coreParam)
                    {
                        return pParam->index();
                    }
                }
            }
            else if (cls == PARAM_OUTPUT && pGroup->data(ROLE_VPARAM_NAME).toString() == "Out Sockets")
            {
                for (int k = 0; k < pGroup->rowCount(); k++)
                {
                    VParamItem* pParam = static_cast<VParamItem*>(pGroup->child(k));
                    if (pParam->data(ROLE_PARAM_NAME) == coreParam)
                    {
                        return pParam->index();
                    }
                }
            }
        }
    }
    return QModelIndex();
}

VPARAM_INFO ViewParamModel::exportParams() const
{
    VPARAM_INFO root;
    QStandardItem* rootItem = invisibleRootItem()->child(0);
    if (!rootItem)
        return root;

    root.m_cls = PARAM_UNKNOWN;
    root.vType = VPARAM_ROOT;
    for (int i = 0; i < rootItem->rowCount(); i++)
    {
        QStandardItem* pTab = rootItem->child(i);

        VPARAM_INFO tab;
        tab.m_cls = PARAM_UNKNOWN;
        tab.vType = VPARAM_TAB;
        tab.m_info.name = pTab->data(ROLE_VPARAM_NAME).toString();

        for (int j = 0; j < pTab->rowCount(); j++)
        {
            QStandardItem* pGroup = pTab->child(j);
            VPARAM_INFO group;
            group.m_cls = PARAM_UNKNOWN;
            group.vType = VPARAM_GROUP;
            group.m_info.name = pGroup->data(ROLE_VPARAM_NAME).toString();

            for (int k = 0; k < pGroup->rowCount(); k++)
            {
                VParamItem* pParam = static_cast<VParamItem*>(pGroup->child(k));
                VPARAM_INFO param;
                if (group.m_info.name == "In Sockets")
                {
                    param.m_cls = PARAM_INPUT;
                }
                else if (group.m_info.name == "Parameters")
                {
                    param.m_cls = PARAM_PARAM;
                }
                else if (group.m_info.name == "Out Sockets")
                {
                    param.m_cls = PARAM_OUTPUT;
                }
                else
                {
                    param.m_cls = PARAM_UNKNOWN;
                }
                param.vType = VPARAM_PARAM;
                param.controlInfos = pParam->data(ROLE_VPARAM_CTRL_PROPERTIES);
                param.coreParam = pParam->data(ROLE_PARAM_NAME).toString();
                param.m_info = pParam->m_info;

                group.children.append(param);
            }
            tab.children.append(group);
        }
        root.children.append(tab);
    }
    return root;
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
                        QModelIndex coreIdx = inputsModel->index(coreparam);
                        paramItem->mapCoreParam(coreIdx);
                    }
                    else if (param.m_cls == PARAM_PARAM)
                    {
                        IParamModel* paramsModel = QVariantPtr<IParamModel>::asPtr(m_nodeIdx.data(ROLE_PARAM_MODEL));
                        QModelIndex coreIdx = paramsModel->index(coreparam);
                        paramItem->mapCoreParam(coreIdx);
                    }
                    else if (param.m_cls == PARAM_OUTPUT)
                    {
                        IParamModel* outputsModel = QVariantPtr<IParamModel>::asPtr(m_nodeIdx.data(ROLE_OUTPUT_MODEL));
                        QModelIndex coreIdx = outputsModel->index(coreparam);
                        paramItem->mapCoreParam(coreIdx);
                    }
                }
                paramItem->m_info = param.m_info;
                paramItem->setData(param.controlInfos, ROLE_VPARAM_CTRL_PROPERTIES);
                if (!coreparam.isEmpty() && (param.m_cls == PARAM_INPUT || param.m_cls == PARAM_OUTPUT))
                {
                    //register subnet param control.
                    const QString &objCls = m_nodeIdx.data(ROLE_OBJNAME).toString();
                    GlobalControlMgr::instance().onParamUpdated(objCls, param.m_cls, coreparam,
                                                                paramItem->m_info.control);
                }
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

    const QString& nodeCls = m_nodeIdx.data(ROLE_OBJNAME).toString();

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
                //PARAM_CONTROL ctrl = (PARAM_CONTROL)idx.data(ROLE_PARAM_CTRL).toInt();
                const QString& typeDesc = idx.data(ROLE_PARAM_TYPE).toString();
                const QVariant& value = idx.data(ROLE_PARAM_VALUE);
                bool bMultiLink = idx.data(ROLE_PARAM_SOCKPROP).toInt() & SOCKPROP_DICTPANEL;
                //until now we can init the control, because control is a "view" property, should be dependent with core data.
                //todo: global control settings, like zfxCode, dict/list panel control, etc.
                //todo: dictpanel should be choosed by custom param manager globally.
                QVariant props;
                PARAM_CONTROL ctrl = CONTROL_NONE;
                if (bMultiLink)
                {
                    ctrl = CONTROL_DICTPANEL;
                }
                else
                {
                    CONTROL_INFO infos = GlobalControlMgr::instance().controlInfo(nodeCls, cls, realName, typeDesc);
                    ctrl = infos.control;
                    props = infos.controlProps;
                }

                VParamItem* paramItem = new VParamItem(VPARAM_PARAM, displayName, true);
                paramItem->m_info.control = ctrl;
                paramItem->m_info.typeDesc = typeDesc;
                paramItem->m_info.value = value;
                paramItem->mapCoreParam(idx);
                paramItem->setData(true, ROLE_VAPRAM_EDITTABLE);
                paramItem->setData(props, ROLE_VPARAM_CTRL_PROPERTIES);
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
                const QString& typeDesc = idx.data(ROLE_PARAM_TYPE).toString();
                const QString& displayName = realName;  //todo: mapping name.
                const QVariant& value = idx.data(ROLE_PARAM_VALUE);

                VParamItem* paramItem = new VParamItem(VPARAM_PARAM, displayName, true);

                CONTROL_INFO infos = GlobalControlMgr::instance().controlInfo(nodeCls, cls, realName, typeDesc);

                paramItem->m_info.control = infos.control;
                paramItem->mapCoreParam(idx);
                paramItem->setData(true, ROLE_VAPRAM_EDITTABLE);
                paramItem->setData(infos.controlProps, ROLE_VPARAM_CTRL_PROPERTIES);
                paramItem->m_info.typeDesc = typeDesc;
                paramItem->m_info.value = value;
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
                const QString& typeDesc = idx.data(ROLE_PARAM_TYPE).toString();
                const QString& displayName = realName;  //todo: mapping name.
                const QVariant& value = idx.data(ROLE_PARAM_VALUE);

                PARAM_CONTROL ctrl = CONTROL_NONE;
                VParamItem* paramItem = new VParamItem(VPARAM_PARAM, displayName, true);
                paramItem->m_info.control = ctrl;
                paramItem->mapCoreParam(idx);
                paramItem->setData(true, ROLE_VAPRAM_EDITTABLE);
                paramItem->m_info.typeDesc = typeDesc;
                paramItem->m_info.value = value;
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
        case ROLE_INPUT_MODEL:
        case ROLE_PARAM_MODEL:
        case ROLE_OUTPUT_MODEL:
            if (m_nodeIdx.isValid())
                return m_nodeIdx.data(role);
            break;
        case ROLE_NODE_IDX:
            return m_nodeIdx;
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
