#pragma once


#include <zs/editor/UI/Widget.h>


namespace zeno2::UI {


struct GraphicsWidget : Widget {
    bool selected = false;
    bool selectable = false;
    bool draggable = false;
};


}  // namespace zeno2::UI
