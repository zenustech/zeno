#ifndef __ZENO_SUBNET_LISTVIEW_H__
#define __ZENO_SUBNET_LISTVIEW_H__

#include <QtWidgets>

class GraphsModel;
class GraphsPlainModel;
class IGraphsModel;
class ZenoSubnetTreeView;

class ZSubnetListModel : public QStandardItemModel
{
    Q_OBJECT
public:
    ZSubnetListModel(IGraphsModel* pModel, QObject* parent = nullptr);
    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    QModelIndex index(int row, int column, const QModelIndex& parent = QModelIndex()) const override;

private:
    GraphsModel* m_model;
};

class ZenoSubnetListView : public QListView
{
    Q_OBJECT
public:
    ZenoSubnetListView(QWidget* parent = nullptr);
    ~ZenoSubnetListView();
    void initModel(IGraphsModel* pModel);
    QSize sizeHint() const override;
    void edittingNew();

signals:
    void graphToBeActivated(const QString&);

protected slots:
    void closeEditor(QWidget* editor, QAbstractItemDelegate::EndEditHint hint) override;

protected:
    void paintEvent(QPaintEvent* e) override;
};

#endif