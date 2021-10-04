#pragma once


#include <z2/UI/Widget.h>


namespace z2::UI {


struct GraphicsWidget : Widget {
    bool selected = false;
    bool selectable = false;
    bool draggable = false;

    virtual void on_position_changed();
};


}  // namespace z2::UI
