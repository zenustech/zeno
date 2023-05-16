#include "zdicttableview.h"


ZDictTableView::ZDictTableView(QWidget* parent)
    : QTableView(parent)
{
    horizontalHeader()->setStretchLastSection(true);
}

void ZDictTableView::keyReleaseEvent(QKeyEvent* event)
{
    if (event->key() == Qt::Key_Delete)
    {
        QModelIndexList lst = this->selectionModel()->selectedRows();
        if (!lst.isEmpty())
        {
            model()->removeRow(lst[0].row());
        }
    }
    return QTableView::keyReleaseEvent(event);
}