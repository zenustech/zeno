#ifndef __GRAPHICS_PLAIN_MODEL_H__
#define __GRAPHICS_PLAIN_MODEL_H__

#include <QtWidgets>

class GraphsModel;
class IGraphsModel;

class GraphsPlainModel : public QStandardItemModel
{
	Q_OBJECT
public:
	GraphsPlainModel(QObject* parent = nullptr);
	~GraphsPlainModel();
	void init(IGraphsModel* pModel);
	bool setData(const QModelIndex& index, const QVariant& value, int role = Qt::EditRole) override;
	void submit(const QModelIndex& idx);
	void revert(const QModelIndex& idx);

public slots:
	void on_dataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles = QVector<int>());
	void on_rowsAboutToBeInserted(const QModelIndex& parent, int first, int last);
	void on_rowsInserted(const QModelIndex& parent, int first, int last);
	void on_rowsAboutToBeRemoved(const QModelIndex& parent, int first, int last);
	void on_rowsRemoved(const QModelIndex& parent, int first, int last);
	bool submit() override;
	void revert() override;

private:
	GraphsModel* m_model;
};


#endif