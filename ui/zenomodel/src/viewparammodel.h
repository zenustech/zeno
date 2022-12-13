#ifndef __VIEW_PARAM_MODEL_H__
#define __VIEW_PARAM_MODEL_H__

#include <QtWidgets>
#include "modeldata.h"
#include <zenomodel/include/jsonhelper.h>

#include <rapidxml/rapidxml_print.hpp>

#define ENABLE_DRAG_DROP_ITEM

using namespace rapidxml;

struct VParamItem;
class IGraphsModel;

class ProxySlotObject : public QObject
{
    Q_OBJECT
public:
    ProxySlotObject(VParamItem* pItem, QObject* parent = nullptr);
    ~ProxySlotObject();
    void mapCoreIndex(const QPersistentModelIndex& idx);
    void unmap();

public slots:
    void onDataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles);

private:
    VParamItem* m_pItem;
};


struct VParamItem : public QStandardItem
{
    QPersistentModelIndex m_index;      //index to core param, see IParamModel.

    //for easy to debug, store here rather than QStandardItem internal data:
    PARAM_INFO m_info;
    VPARAM_TYPE vType;
    VPARAM_INFO m_tempInfo;

    ProxySlotObject m_proxySlot;

    VParamItem(VPARAM_TYPE vType, const QString& text, bool bMapCore = false);
    VParamItem(VPARAM_TYPE vType, const QIcon& icon, const QString& text, bool bMapCore = false);

    VParamItem(const VParamItem& other);
    ~VParamItem();

    QVariant data(int role = Qt::UserRole + 1) const override;
    void setData(const QVariant& value, int role) override;
    QStandardItem* clone() const override;
    void mapCoreParam(const QPersistentModelIndex& idx);
    rapidxml::xml_node<>* exportXml(rapidxml::xml_document<>& doc);
    VParamItem* getItem(const QString& uniqueName) const;
    bool operator==(VParamItem* rItem) const;
#ifdef ENABLE_DRAG_DROP_ITEM
    void read(QDataStream& in) override;
    void write(QDataStream& out) const override;
#endif
};

class ViewParamModel : public QStandardItemModel
{
    Q_OBJECT
public:
    explicit ViewParamModel(bool bNodeUI, const QModelIndex& nodeIdx, IGraphsModel* pModel, QObject* parent = nullptr);
    ~ViewParamModel();
    void clone(ViewParamModel* pModel);
    QPersistentModelIndex nodeIdx() const;
    QModelIndex indexFromPath(const QString& path);
    QModelIndex indexFromName(PARAM_CLASS cls, const QString& coreParam);
    void resetParams(const VPARAM_INFO& invisibleRoot);
    VPARAM_INFO exportParams() const;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    bool isNodeModel() const;
    bool isDirty() const;
    void markDirty();
#ifdef ENABLE_DRAG_DROP_ITEM
    QMimeData* mimeData(const QModelIndexList& indexes) const override;
    bool dropMimeData(const QMimeData* data, Qt::DropAction action, int row, int column, const QModelIndex& parent) override;
    Qt::ItemFlags flags(const QModelIndex &index) const override;
#endif

public slots:
    void onCoreParamsInserted(const QModelIndex& parent, int first, int last);
    void onCoreParamsAboutToBeRemoved(const QModelIndex& parent, int first, int last);

private:
    void setup(const QString& customUI);
    void initCustomUI();

    const QPersistentModelIndex m_nodeIdx;
    const IGraphsModel* const m_model;
    const bool m_bNodeUI;
    bool m_bDirty;
};

#endif