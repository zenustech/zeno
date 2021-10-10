#pragma once


#include <zs/editor/UI/GraphicsWidget.h>


namespace zs::editor::UI {


struct GraphicsRectItem : GraphicsWidget {
    void paint() const override;
};


}  // namespace zs::editor::UI
