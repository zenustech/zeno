#pragma once


#include <z2/UI/GraphicsView.h>


namespace z2::UI {


struct UiVisViewport : GraphicsView {
    void do_paint() override;
    void paint() const override;
};


}
