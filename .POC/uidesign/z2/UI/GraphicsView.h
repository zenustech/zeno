#pragma once


#include <z2/UI/GraphicsWidget.h>


namespace z2::UI {


struct GraphicsView : Widget {
    std::set<GraphicsWidget *> children_selected;

    Point translate = {0, 0};
    float scaling = 1.f;

    virtual void select_child(GraphicsWidget *ptr, bool multiselect);
    ztd::dtor_function do_transform() const override;
    void on_event(Event_Motion e) override;
    void on_event(Event_Mouse e) override;
    void on_event(Event_Scroll e) override;
    void do_paint() override;
};


}  // namespace z2::UI
