#ifndef __VIEW_PARAM_MODEL_H__
#define __VIEW_PARAM_MODEL_H__

#include <QtWidgets>
#include "modeldata.h"
#include <zenomodel/include/jsonhelper.h>

#include <rapidxml/rapidxml_print.hpp>

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
};

class ViewParamModel : public QStandardItemModel
{
    Q_OBJECT
public:
    explicit ViewParamModel(bool bNodeUI, const QModelIndex& nodeIdx, IGraphsModel* pModel, QObject* parent = nullptr);
    void clone(ViewParamModel* pModel);
    QPersistentModelIndex nodeIdx() const;
    void resetParams(const VPARAM_INFO& invisibleRoot);
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    QString exportXml();
    bool isNodeModel() const;

public slots:
    void onCoreParamsInserted(const QModelIndex& parent, int first, int last);
    void onCoreParamsAboutToBeRemoved(const QModelIndex& parent, int first, int last);

private:
    void setup(const QString& customUI);
    void initPanel();
    void initNode();

    const bool m_bNodeUI;
    const QPersistentModelIndex m_nodeIdx;
    const IGraphsModel* const m_model;
};

#endif