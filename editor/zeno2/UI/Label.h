#pragma once


#include <zeno2/UI/Widget.h>
#include <zeno2/UI/Font.h>


namespace zeno2::UI {


struct Label : Widget {
    static constexpr float BW = 8.f;

    Label();

    std::string text;

    float font_size = 20.f;
    FTGL::TextAlignment alignment = FTGL::ALIGN_LEFT;

    void paint() const override;
};


}  // namespace zeno2::UI
