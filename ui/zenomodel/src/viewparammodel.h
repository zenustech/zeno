#ifndef __VIEW_PARAM_MODEL_H__
#define __VIEW_PARAM_MODEL_H__

#include <QtWidgets>
#include "modeldata.h"
#include <zenomodel/include/jsonhelper.h>
#include <rapidxml/rapidxml_print.hpp>

using namespace rapidxml;

class IGraphsModel;
struct VParamItem;

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

    //for node ui params:
    void getNodeParams(QModelIndexList& inputs, QModelIndexList& params, QModelIndexList& outputs);
    QModelIndexList paramsIndice();
    QModelIndexList outputsIndice();
    void arrangeOrder(const QStringList &inputKeys, const QStringList &outputKeys);

    static VParamItem* importParam(const VPARAM_INFO& param);
    void importParamInfo(const VPARAM_INFO& invisibleRoot);
    VPARAM_INFO exportParams() const;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    bool isNodeModel() const;
    bool isDirty() const;
    void markDirty();
    QMimeData* mimeData(const QModelIndexList& indexes) const override;
    bool dropMimeData(const QMimeData* data, Qt::DropAction action, int row, int column, const QModelIndex& parent) override;
    Qt::ItemFlags flags(const QModelIndex &index) const override;
    bool moveRows(const QModelIndex& sourceParent, int sourceRow, int count, const QModelIndex& destinationParent,
                  int destinationChild) override;

signals:
    void editNameChanged(const QModelIndex& itemIdx, const QString& oldPath, const QString& newName);

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