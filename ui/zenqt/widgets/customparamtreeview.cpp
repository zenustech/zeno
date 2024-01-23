#include "customparamtreeview.h"


CustomParamTreeView::CustomParamTreeView(QWidget* parent)
    : _base(parent)
{
}

void CustomParamTreeView::dragMoveEvent(QDragMoveEvent* event)
{
    _base::dragMoveEvent(event);
}

void CustomParamTreeView::dropEvent(QDropEvent* event)
{
    _base::dropEvent(event);
}