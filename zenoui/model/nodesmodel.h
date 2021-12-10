#ifndef __ZENO_NODES_MODEL_H__
#define __ZENO_NODES_MODEL_H__

#include <QModelIndex>
#include <QString>
#include <QObject>
#include "nodeitem.h"

class NodesModel : public QAbstractItemModel
{
    Q_OBJECT
    typedef QAbstractItemModel _base;
public:
	explicit NodesModel(QObject* parent = nullptr);
	~NodesModel();

	//QAbstractItemModel
	QModelIndex index(int row, int column, const QModelIndex& parent = QModelIndex()) const override;

	QModelIndex parent(const QModelIndex& child) const override;
	int rowCount(const QModelIndex& parent = QModelIndex()) const override;
	int columnCount(const QModelIndex& parent = QModelIndex()) const override;
    bool hasChildren(const QModelIndex &parent = QModelIndex()) const override;

	QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
	bool setData(const QModelIndex& index, const QVariant& value, int role = Qt::EditRole) override;

	QVariant headerData(int section, Qt::Orientation orientation,
                        int role = Qt::DisplayRole) const override;
    bool setHeaderData(int section, Qt::Orientation orientation, const QVariant &value,
                       int role = Qt::EditRole) override;
    QMap<int, QVariant> itemData(const QModelIndex &index) const override;
    bool setItemData(const QModelIndex &index, const QMap<int, QVariant> &roles) override;

	QModelIndexList match(const QModelIndex &start, int role,
                          const QVariant &value, int hits = 1,
                          Qt::MatchFlags flags =
                          Qt::MatchFlags(Qt::MatchStartsWith | Qt::MatchWrap)) const override;
    QHash<int, QByteArray> roleNames() const override;


	//NodeItemModel
    QModelIndex index(QString id, const QModelIndex &parent = QModelIndex()) const;
    SP_NODE_ITEM itemFromIndex(const QModelIndex &index) const;
    void appendItem(SP_NODE_ITEM pItem);
    bool insertRow(int row, SP_NODE_ITEM pItem, const QModelIndex &parent = QModelIndex());
    bool removeRow(int row, const QModelIndex &parent = QModelIndex());
    SP_NODE_ITEM rootItem() const;

private:
    SP_NODE_ITEM m_rootItem;
};

#endif
