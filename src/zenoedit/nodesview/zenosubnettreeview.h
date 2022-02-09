#ifndef __ZENO_SUBNET_TREEVIEW_H__
#define __ZENO_SUBNET_TREEVIEW_H__

#include <QtWidgets>

class GraphsModel;
class GraphsTreeModel;

class ZenoSubnetTreeView : public QTreeView
{
    Q_OBJECT
public:
    ZenoSubnetTreeView(QWidget* parent = nullptr);
    ~ZenoSubnetTreeView();
    void initModel(GraphsTreeModel* pModel);

protected:
    void paintEvent(QPaintEvent* e) override;

private:
    void initStyle();
};

#endif