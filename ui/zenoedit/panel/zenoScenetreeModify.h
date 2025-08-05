#pragma once

#include <QtWidgets>
#include <QAbstractTableModel>
#include <tinygltf/json.hpp>

using Json = nlohmann::json;

class ResetIconDelegate : public QStyledItemDelegate {
	Q_OBJECT

public:
	explicit ResetIconDelegate(QObject* parent = nullptr);

	void paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const override;
};

class scenetreeModifyModel : public QAbstractTableModel {
	Q_OBJECT

public:
	explicit scenetreeModifyModel(QObject* parent = nullptr);
	~scenetreeModifyModel();

	int rowCount(const QModelIndex& parent = QModelIndex()) const override;
	int columnCount(const QModelIndex& parent = QModelIndex()) const override;
	QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;

	QVariant headerData(int section, Qt::Orientation orientation, int role) const override;
	bool setData(const QModelIndex& index, const QVariant& value, int role = Qt::EditRole) override;

	void insertRow(const QString id, const QString r0, const QString r1, const QString r2, const QString t);
	void removeRow(int row);
	QModelIndex& indexFromId(QString id);
	std::vector<std::string> getRow(int row) const;
	void setupModelDataFromMessage(Json const& content);

private:
	struct ModifyItem {
		QString id;
		QString r0;
		QString r1;
		QString r2;
		QString t;
	};
	QVector<ModifyItem> m_items;
};

class zenoScenetreeModify  : public QWidget
{
	Q_OBJECT

public:
	zenoScenetreeModify(QWidget *parent = nullptr);
	~zenoScenetreeModify();

private:
	void initUi();
	void generateModificationNode(QString outNodeId, QString outSock, QString inNodeType, QString inSock, QString inModifyInfoSock, Json& msg);

	QTableView* m_tableView = nullptr;
	scenetreeModifyModel* m_model = nullptr;
};

