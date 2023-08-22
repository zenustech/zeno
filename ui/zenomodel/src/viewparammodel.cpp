#include "viewparammodel.h"
#include "parammodel.h"
#include "zassert.h"
#include "modelrole.h"
#include <zenomodel/include/uihelper.h>
#include "variantptr.h"
#include <zenomodel/customui/customuirw.h>
#include "globalcontrolmgr.h"
#include "vparamitem.h"
#include "common_def.h"
#include "iotags.h"


ViewParamModel::ViewParamModel(const QModelIndex& nodeIdx, IGraphsModel* pModel, QObject* parent)
    : QStandardItemModel(parent)
    , m_nodeIdx(nodeIdx)
    , m_pGraphsModel(pModel)
{
    setItemPrototype(new VParamItem(VPARAM_PARAM, ""));
}

ViewParamModel::~ViewParamModel()
{
}

QModelIndex ViewParamModel::indexFromPath(const QString& path)
{
    QString root, tab, group, param;

    QStringList lst = path.split("/", QtSkipEmptyParts);
    if (lst.isEmpty())
        return QModelIndex();

    if (lst.size() >= 1)
        root = lst[0];
    if (lst.size() >= 2)
        tab = lst[1];
    if (lst.size() >= 3)
        group = lst[2];
    if (lst.size() >= 4)
        param = lst[3];

    if (root != "root") return QModelIndex();

    QStandardItem* rootItem = invisibleRootItem()->child(0);
    if (!rootItem)
        return QModelIndex();

    if (lst.size() == 1)
        return rootItem->index();

    for (int i = 0; i < rootItem->rowCount(); i++)
    {
        QStandardItem* pTab = rootItem->child(i);
        if (pTab->data(ROLE_VPARAM_NAME) == tab)
        {
            if (lst.size() == 2)
                return pTab->index();

            for (int j = 0; j < pTab->rowCount(); j++)
            {
                QStandardItem* pGroup = pTab->child(j);
                if (pGroup->data(ROLE_VPARAM_NAME) == group)
                {
                    if (lst.size() == 3)
                        return pGroup->index();

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

bool ViewParamModel::canDropMimeData(const QMimeData* data, Qt::DropAction action, int row, int column, const QModelIndex& parent) const
{
    const QString& groupname = parent.data(ROLE_PARAM_NAME).toString();
    const QString& movingItemGroup = data->data("group");
    //zeno::log_info("groupname: {}", groupname.toStdString());
    //zeno::log_info("row: {}", row);
    if (groupname.isEmpty() || movingItemGroup != groupname || row < 0 || !parent.isValid()) {
        return false;
    }
    return QStandardItemModel::canDropMimeData(data, action, row, column, parent);
}

QMimeData* ViewParamModel::mimeData(const QModelIndexList &indexes) const
{
    QMimeData* mimeD = new QMimeData();// QAbstractItemModel::mimeData(indexes);
    if (indexes.size() > 0)
    {
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

        QStandardItem* parentItem = node->parent();
        if (parentItem)
        {
            const QString& groupName = parentItem->data(ROLE_PARAM_NAME).toString();
            mimeD->setData("group", groupName.toUtf8());
        }
    }
    else
    {
        mimeD->setData("Node/NodePtr", "NULL");
    }
    return mimeD;
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

bool ViewParamModel::moveRows(
        const QModelIndex& sourceParent,
        int srcRow,
        int count,
        const QModelIndex& destinationParent,
        int dstRow)
{
    if (count != 1)
        return false;       //todo: multiline movement.

    VParamItem* srcParent = static_cast<VParamItem*>(itemFromIndex(sourceParent));
    if (!srcParent)
        return false;

    if (!srcParent->child(srcRow))
        return false;

    QStandardItem *dstParent = itemFromIndex(destinationParent);
    //only allow the movement in the same layer.
    if (!dstParent || srcParent != dstParent)
        return false;

    if (srcParent == dstParent && srcRow == dstRow && count == 1)
        return false;

    QStandardItem* srcItem = nullptr;

    QPersistentModelIndex sourceRowIdx = index(srcRow, 0, sourceParent);

    beginMoveRows(sourceParent, srcRow, srcRow, destinationParent, dstRow);
    {
        BlockSignalScope scope(this);

        //cannot insert and then remove item, because the persist indice will be destroyed.
        /*
            QStandardItem* srcItem = srcParent->takeChild(srcRow);
            dstParent->insertRow(dstRow, srcItem);
            srcParent->removeRow(srcRow + 1);
        */
        VParamItem* pSrcItem = static_cast<VParamItem*>(srcParent->child(srcRow));
        VParamItem* pSrcItemClone = new VParamItem(VPARAM_PARAM, pSrcItem->m_name);
        pSrcItemClone->cloneFrom(pSrcItem);
        if (srcRow < dstRow)
        {
            for (int r = srcRow + 1; r <= dstRow; r++)
            {
                //copy data from data[r] to data[r-1].
                VParamItem* pItem_r = static_cast<VParamItem*>(srcParent->child(r));
                VParamItem* pItem_r_minus_1 = static_cast<VParamItem *>(srcParent->child(r - 1));
                pItem_r_minus_1->cloneFrom(pItem_r);
            }
        }
        else
        {
            for (int r = srcRow - 1; r >= dstRow; r--)
            {
                //copy data from row r to row r + 1;
                VParamItem* pItem_r = static_cast<VParamItem*>(srcParent->child(r));
                VParamItem* pItem_r_plus_1 = static_cast<VParamItem *>(srcParent->child(r + 1));
                pItem_r_plus_1->cloneFrom(pItem_r);
            }
        }
        VParamItem *pDstItem = static_cast<VParamItem *>(srcParent->child(dstRow));
        pDstItem->cloneFrom(pSrcItemClone);
    }
    endMoveRows();
    return true;
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
    const QString& coreparam = pItem->m_tempInfo.refParamPath;
    pItem->setData(pItem->m_tempInfo.m_info.control, ROLE_PARAM_CTRL);
    pItem->setData(pItem->m_tempInfo.m_info.typeDesc, ROLE_PARAM_TYPE);
    pItem->setData(pItem->m_tempInfo.m_info.name, ROLE_PARAM_NAME);
    pItem->setText(pItem->m_tempInfo.m_info.name);
    pItem->setData(pItem->m_tempInfo.m_info.value, ROLE_PARAM_VALUE);
    pItem->m_uuid = pItem->m_tempInfo.m_uuid;

    if (!coreparam.isEmpty())
    {
        QModelIndex refIdx = m_pGraphsModel->indexFromPath(coreparam);
        pItem->mapCoreParam(refIdx);
        pItem->setData(true, ROLE_VPARAM_IS_COREPARAM);
        //copy inner type and value.
        if (pItem->m_index.isValid())
        {
            pItem->setData(pItem->m_index.data(ROLE_PARAM_TYPE), ROLE_PARAM_TYPE);
            pItem->setData(pItem->m_index.data(ROLE_PARAM_VALUE), ROLE_PARAM_VALUE);
        }
    }
    else
    {
        pItem->setData(false, ROLE_VPARAM_IS_COREPARAM);
    }
    
    pItem->setData(pItem->m_tempInfo.controlInfos, ROLE_VPARAM_CTRL_PROPERTIES);
    pItem->setData(pItem->m_tempInfo.m_info.toolTip, ROLE_VPARAM_TOOLTIP);
    pItem->m_sockProp = pItem->m_tempInfo.m_info.sockProp;
    return true;
}

VPARAM_INFO ViewParamModel::exportParams() const
{
    VParamItem* rootItem = static_cast<VParamItem*>(invisibleRootItem()->child(0));
    if (!rootItem)
        return VPARAM_INFO();
    return rootItem->exportParamInfo();
}

QPersistentModelIndex ViewParamModel::nodeIdx() const
{
    return m_nodeIdx;
}

IGraphsModel* ViewParamModel::graphsModel() const
{
    return m_pGraphsModel;
}

QVariant ViewParamModel::data(const QModelIndex& index, int role) const
{
    switch (role)
    {
        case ROLE_OBJID:    return m_nodeIdx.isValid() ? m_nodeIdx.data(role) : "";
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

bool ViewParamModel::isDirty() const
{
    return false;
}

void ViewParamModel::markDirty()
{
}

void ViewParamModel::clone(ViewParamModel* pModel)
{
    QStandardItem* pRoot = invisibleRootItem();
    ZASSERT_EXIT(pRoot);
    while (pRoot->rowCount() > 0)
        pRoot->removeRow(0);

    QStandardItem* pRightRoot = pModel->invisibleRootItem();
    for (int r = 0; r < pRightRoot->rowCount(); r++)
    {
        QStandardItem* newItem = pRightRoot->child(r)->clone();
        pRoot->appendRow(newItem);
    }
}

bool ViewParamModel::isEditable(const QModelIndex &current) 
{
    return false;
}
