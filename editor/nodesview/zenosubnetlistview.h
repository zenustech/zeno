#ifndef __ZENO_SUBNET_LISTVIEW_H__
#define __ZENO_SUBNET_LISTVIEW_H__

#include <QtWidgets>

class GraphsModel;

class ZSubnetListModel : public QStandardItemModel
{
    Q_OBJECT
public:
    ZSubnetListModel(GraphsModel* pModel, QObject* parent = nullptr);
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
    void initModel(GraphsModel* pModel);
    QSize sizeHint() const override;

protected:
    void paintEvent(QPaintEvent* e) override;
};

class ZenoSubnetListPanel : public QWidget
{
    Q_OBJECT
public:
    ZenoSubnetListPanel(QWidget* parent = nullptr);
    void initModel(GraphsModel* pModel);
    QSize sizeHint() const override;

signals:
    void clicked(const QModelIndex& index);

private slots:
    void onNewSubnetBtnClicked();
    void onModelReset();

private:
    ZenoSubnetListView* m_pListView;
    QLabel* m_pTextLbl;
};

#endif