#ifndef __DICTKEY_MODEL_H__
#define __DICTKEY_MODEL_H__

#include <QtWidgets>

class IGraphsModel;

struct _DictItem
{
    QString key;
    QList<QPersistentModelIndex> links;
};

class DictKeyModel : public QAbstractItemModel
{
    Q_OBJECT
public:
    DictKeyModel(IGraphsModel* pGraphs, const QModelIndex& dictParam, QObject* parent = nullptr);
    ~DictKeyModel();
    void clearAll();

    QModelIndex index(int row, int column, const QModelIndex& parent = QModelIndex()) const override;
    QModelIndex index(const QString &keyName);
    QModelIndex parent(const QModelIndex& child) const override;
    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    int columnCount(const QModelIndex& parent = QModelIndex()) const override;

    QVariant data(const QModelIndex& index, int role) const override;
    bool setData(const QModelIndex& index, const QVariant& value, int role = Qt::EditRole) override;

    bool insertRows(int row, int count, const QModelIndex& parent = QModelIndex()) override;
    bool insertColumns(int column, int count, const QModelIndex& parent = QModelIndex()) override;
    bool removeRows(int row, int count, const QModelIndex &parent = QModelIndex()) override;
    bool removeColumns(int column, int count, const QModelIndex &parent = QModelIndex()) override;
    bool moveRows(const QModelIndex &sourceParent, int sourceRow, int count, const QModelIndex &destinationParent,
                  int destinationChild) override;
    bool moveColumns(const QModelIndex &sourceParent, int sourceColumn, int count,
                   const QModelIndex &destinationParent, int destinationChild) override;
    bool isCollasped() const;
    void setCollasped(bool bCollasped);

private:
    QPersistentModelIndex m_dictParam;   //core param of dict param.
    QVector<_DictItem> m_items;
    IGraphsModel* m_pGraphs;
    bool m_bCollasped;
};

#endif