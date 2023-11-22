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
    explicit ViewParamModel(const QModelIndex& nodeIdx, IGraphsModel* pModel, QObject* parent = nullptr);
    ~ViewParamModel();
    virtual void clone(ViewParamModel* pModel);
    IGraphsModel* graphsModel() const;
    virtual QModelIndex indexFromPath(const QString& path);
    QModelIndex indexFromName(PARAM_CLASS cls, const QString& coreParam);

    VPARAM_INFO exportParams() const;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    virtual bool isDirty() const;
    virtual void markDirty();
    QMimeData* mimeData(const QModelIndexList& indexes) const override;
    bool canDropMimeData(const QMimeData* data, Qt::DropAction action, int row, int column, const QModelIndex& parent) const override;
    bool dropMimeData(const QMimeData* data, Qt::DropAction action, int row, int column, const QModelIndex& parent) override;
    Qt::ItemFlags flags(const QModelIndex &index) const override;
    bool moveRows(const QModelIndex& sourceParent, int sourceRow, int count, const QModelIndex& destinationParent,
                  int destinationChild) override;
    virtual bool isEditable(const QModelIndex &current);

signals:
    void editNameChanged(const QModelIndex& itemIdx, const QString& oldPath, const QString& newName);

protected:
    IGraphsModel* m_pGraphsModel;
};

#endif