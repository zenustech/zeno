#include "zenosubnettreeview.h"
#include <zenoui/model/graphsmodel.h>
#include <model/graphstreemodel.h>


ZenoSubnetTreeView::ZenoSubnetTreeView(QWidget* parent)
    : QTreeView(parent)
{
    header()->setVisible(false);
}

ZenoSubnetTreeView::~ZenoSubnetTreeView()
{
}

void ZenoSubnetTreeView::initModel(GraphsTreeModel* pModel)
{
    setModel(pModel);
}

void ZenoSubnetTreeView::paintEvent(QPaintEvent* e)
{
    QTreeView::paintEvent(e);
}
