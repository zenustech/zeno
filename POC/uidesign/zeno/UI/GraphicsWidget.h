#pragma once


#include <zeno/UI/Widget.h>


struct GraphicsWidget : Widget {
    bool selected = false;
    bool selectable = false;
    bool draggable = false;
};


struct GraphicsView : Widget {
    std::set<GraphicsWidget *> children_selected;

    virtual void select_child(GraphicsWidget *ptr, bool multiselect);
    void on_event(Event_Motion e) override;
    void on_event(Event_Mouse e) override;
};
