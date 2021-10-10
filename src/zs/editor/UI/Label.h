#pragma once


#include <zs/editor/UI/Widget.h>
#include <zs/editor/UI/Font.h>


namespace zs::editor::UI {


struct Label : Widget {
    static constexpr float BW = 8.f;

    Label();

    std::string text;

    float font_size = 20.f;
    FTGL::TextAlignment alignment = FTGL::ALIGN_LEFT;

    void paint() const override;
};


}  // namespace zs::editor::UI
