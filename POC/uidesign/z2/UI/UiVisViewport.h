#pragma once


#include <z2/UI/GraphicsView.h>
#include <z2/GL/Camera.h>


namespace z2::UI {


struct UiMainWindow;


struct UiVisViewport : GraphicsView {
    GL::Camera camera;

    void do_paint() override;
    void paint() const override;

    inline UiMainWindow *get_parent() const {
        return (UiMainWindow *)parent;
    }
};


}
