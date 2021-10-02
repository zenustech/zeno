#include "GraphicsWidget.h"


void GraphicsView::select_child(GraphicsWidget *ptr, bool multiselect) {
    if (!(multiselect || (ptr && ptr->selected))) {
        for (auto const &child: children_selected) {
            child->selected = false;
        }
        children_selected.clear();
    }
    if (ptr) {
        if (ptr->selected && multiselect) {
            children_selected.erase(ptr);
            ptr->selected = false;
        } else {
            children_selected.insert(ptr);
            ptr->selected = true;
        }
    }
}

void GraphicsView::on_event(Event_Motion e) {
    Widget::on_event(e);
    if (cur.lmb) {
        for (auto const &child: children_selected) {
            if (child->draggable) {
                child->position.x += cur.dx;
                child->position.y += cur.dy;
            }
        }
    }
}

void GraphicsView::on_event(Event_Mouse e) {
    Widget::on_event(e);

    if (e.down != true)
        return;
    if (e.btn != 0)
        return;

    if (auto item = item_at({cur.x, cur.y}); item) {
        auto it = dynamic_cast<GraphicsWidget *>(item);
        if (it && it->selectable)
            select_child(it, cur.shift);
    } else if (!cur.shift) {
        select_child(nullptr, false);
    }
}
