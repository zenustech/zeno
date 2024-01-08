#ifndef __PARAMS_MODEL_H__
#define __PARAMS_MODEL_H__

#include <QObject>
#include <QStandardItemModel>
#include <QQuickItem>
#include "common.h"

struct ParamItem
{
    bool bInput = true;
    QString name;
    QString type;
    ParamControl::Value control = ParamControl::None;
    QList<QPersistentModelIndex> links;
};

class ParamsModel : public QAbstractListModel
{
    Q_OBJECT
    QML_ELEMENT

public:
    ParamsModel(NODE_DESCRIPTOR desc, QObject* parent = nullptr);

    Q_INVOKABLE int indexFromName(const QString& name, bool bInput) const;
    Q_INVOKABLE QVariant getIndexList(bool bInput) const;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    bool setData(const QModelIndex& index, const QVariant& value, int role = Qt::EditRole) override;
    QHash<int, QByteArray> roleNames() const override;
    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    bool removeRows(int row, int count, const QModelIndex& parent = QModelIndex()) override;

    //api:
    void setNodeIdx(const QModelIndex& nodeIdx);
    QModelIndex paramIdx(const QString& name, bool bInput) const;
    void addLink(const QModelIndex& paramIdx, const QPersistentModelIndex& linkIdx);
    int removeLink(const QModelIndex& paramIdx);
    void addParam(const ParamItem& param);

    int getParamlinkCount(const QModelIndex& paramIdx);

private:
    QPersistentModelIndex m_nodeIdx;
    QVector<ParamItem> m_items;
};


#endif