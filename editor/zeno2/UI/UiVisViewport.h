#pragma once


#include <zeno2/UI/GraphicsView.h>
#include <zeno2/GL/Camera.h>


namespace zeno2::UI {


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
