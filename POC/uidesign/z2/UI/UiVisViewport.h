#pragma once


#include <z2/UI/GraphicsView.h>


namespace z2::UI {


struct UiMainWindow;


struct UiVisViewport : GraphicsView {
    void do_paint() override;
    void paint() const override;

    inline UiMainWindow *get_parent() const {
        return (UiMainWindow *)parent;
    }
};


}
