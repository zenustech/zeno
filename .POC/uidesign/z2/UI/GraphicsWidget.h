#pragma once


#include <z2/UI/Widget.h>


namespace z2::UI {


struct GraphicsWidget : Widget {
    bool selected = false;
    bool selectable = false;
    bool draggable = false;
};


}  // namespace z2::UI
