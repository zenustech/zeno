#pragma once


#include <zs/editor/UI/GraphicsView.h>
#include <zs/editor/GL/Camera.h>


namespace zs::editor::UI {


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
