#ifndef __NODE_ITEM_MODEL_H__
#define __NODE_ITEM_MODEL_H__

#include <QModelIndex>
#include <QString>
#include <QObject>
#include "nodeitem.h"

class NodeItemModel : public QAbstractItemModel
{
    Q_OBJECT
    typedef QAbstractItemModel _base;
public:
	explicit NodeItemModel(QObject* parent = nullptr);
	~NodeItemModel();

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
    NodeItem* itemFromIndex(const QModelIndex &index) const;

private:
	//std::map<QString, NodeItem> datas;
    std::unordered_map<QString, QString> m_name2Id;
    std::unordered_map<QString, NodeItem*> m_idMapprer;
};

#endif
