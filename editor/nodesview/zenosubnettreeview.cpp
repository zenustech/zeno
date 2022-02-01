#include "zenosubnettreeview.h"
#include <model/graphsmodel.h>


ZenoSubnetTreeView::ZenoSubnetTreeView(QWidget* parent)
    : QTreeView(parent)
{
    header()->setVisible(false);
}

ZenoSubnetTreeView::~ZenoSubnetTreeView()
{
}

void ZenoSubnetTreeView::initModel(GraphsModel* pModel)
{
    setModel(pModel);
}

void ZenoSubnetTreeView::paintEvent(QPaintEvent* e)
{
    QTreeView::paintEvent(e);
}