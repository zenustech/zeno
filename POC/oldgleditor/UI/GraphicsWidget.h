#pragma once


#include <zs/editor/UI/Widget.h>


namespace zs::editor::UI {


struct GraphicsWidget : Widget {
    bool selected = false;
    bool selectable = false;
    bool draggable = false;
};


}  // namespace zs::editor::UI
