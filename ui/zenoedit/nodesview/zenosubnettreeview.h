#ifndef __ZENO_SUBNET_TREEVIEW_H__
#define __ZENO_SUBNET_TREEVIEW_H__

#include <QtWidgets>

class GraphsTreeModel;
class IGraphsModel;

class ZenoSubnetTreeView : public QTreeView
{
    Q_OBJECT
public:
    ZenoSubnetTreeView(QWidget* parent = nullptr);
    ~ZenoSubnetTreeView();
    void initModel(GraphsTreeModel* pTreeModel);

protected:
    void paintEvent(QPaintEvent* e) override;

private:
    void initStyle();
};

#endif