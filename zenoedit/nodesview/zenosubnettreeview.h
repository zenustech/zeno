#ifndef __ZENO_SUBNET_TREEVIEW_H__
#define __ZENO_SUBNET_TREEVIEW_H__

#include <QtWidgets>

class GraphsModel;
class GraphsTreeModel;

class ZenoSubnetTreePanel : public QWidget
{
    Q_OBJECT
public:

};

class ZenoSubnetTreeView : public QTreeView
{
    Q_OBJECT
public:
    ZenoSubnetTreeView(QWidget* parent = nullptr);
    ~ZenoSubnetTreeView();
    void initModel(GraphsTreeModel* pModel);

private:
    void paintEvent(QPaintEvent* e) override;
};

#endif