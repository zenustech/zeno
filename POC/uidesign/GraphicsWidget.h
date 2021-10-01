#pragma once


#include "Widget.h"


struct GraphicsWidget : Widget {
    bool selected = false;
    bool selectable = false;
    bool draggable = false;
};


struct GraphicsView : Widget {
    std::set<GraphicsWidget *> children_selected;

    virtual void select_child(GraphicsWidget *ptr, bool multiselect) {
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

    void on_event(Event_Motion e) override {
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

    void on_event(Event_Mouse e) override {
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
};
