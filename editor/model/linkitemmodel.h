#ifndef __LINK_ITEM_MODEL_H__
#define __LINK_ITEM_MODEL_H__

#include "nodeitem.h"

class LinkItemModel : public QAbstractItemModel
{
	Q_OBJECT
public:
	explicit LinkItemModel(QObject* parent = nullptr);
	~LinkItemModel();

	//QAbstractItemModel
	QModelIndex index(int row, int column, const QModelIndex& parent = QModelIndex()) const override;
	QModelIndex index(QString id, const QModelIndex& parent = QModelIndex()) override;	//custom function, index with id
	QModelIndex parent(const QModelIndex& child) const override;
	int rowCount(const QModelIndex& parent = QModelIndex()) const override;
	int columnCount(const QModelIndex& parent = QModelIndex()) const override;
	QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
	bool setData(const QModelIndex& index, const QVariant& value, int role = Qt::EditRole) override;

	//NodeItemModel


private:
	std::map<QString, LinkItem> datas;
};

#endif