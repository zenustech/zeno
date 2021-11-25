#ifndef __NODE_ITEM_MODEL_H__
#define __NODE_ITEM_MODEL_H__

#include <QModelIndex>
#include <QString>
#include <QOBject>

class NodeItemModel : public QAbstractItemModel
{
	Q_OBJECT
public:
	explicit NodeItemModel(QObject* parent = nullptr);
	~NodeItemModel();

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
	std::map<QString, NodeItem> datas;
};

#endif
