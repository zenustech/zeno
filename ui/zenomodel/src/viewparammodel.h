#ifndef __VIEW_PARAM_MODEL_H__
#define __VIEW_PARAM_MODEL_H__

#include <QtWidgets>
#include <zenomodel/include/modeldata.h>
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
    virtual void clone(ViewParamModel* pModel);
    QPersistentModelIndex nodeIdx() const;
    IGraphsModel* graphsModel() const;
    virtual QModelIndex indexFromPath(const QString& path);
    QModelIndex indexFromName(PARAM_CLASS cls, const QString& coreParam);

    //for node ui params:
    void getNodeParams(QModelIndexList& inputs, QModelIndexList& params, QModelIndexList& outputs);
    void arrangeOrder(const QStringList &inputKeys, const QStringList &outputKeys);

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

    bool isEditable(const QModelIndex &current);

signals:
    void editNameChanged(const QModelIndex& itemIdx, const QString& oldPath, const QString& newName);

protected:
    virtual void initUI();

    const QPersistentModelIndex m_nodeIdx;
    IGraphsModel* m_model;
    const bool m_bNodeUI;
    bool m_bDirty;
};

#endif