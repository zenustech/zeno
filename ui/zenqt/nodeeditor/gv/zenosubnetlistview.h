#ifndef __ZENO_SUBNET_LISTVIEW_H__
#define __ZENO_SUBNET_LISTVIEW_H__

#include <QtWidgets>

class AssetsModel;

class ZenoSubnetListView : public QListView
{
    Q_OBJECT
public:
    ZenoSubnetListView(QWidget* parent = nullptr);
    ~ZenoSubnetListView();
    void initModel(AssetsModel* pModel);
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