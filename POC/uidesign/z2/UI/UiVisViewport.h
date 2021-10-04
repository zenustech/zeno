#pragma once


#include <z2/UI/GraphicsView.h>
#include <z2/GL/Camera.h>


namespace z2::UI {


struct UiMainWindow;


struct UiVisViewport : Widget {
    std::unique_ptr<GL::Camera> camera = std::make_unique<GL::Camera>();

    void do_paint() override;
    void paint() const override;
    void on_event(Event_Motion e) override;
    void on_event(Event_Mouse e) override;
    void on_event(Event_Scroll e) override;

    inline UiMainWindow *get_parent() const {
        return (UiMainWindow *)parent;
    }
};


}
