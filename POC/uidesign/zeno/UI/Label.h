#pragma once


#include <zeno/UI/Widget.h>
#include <zeno/UI/Font.h>


namespace zeno::UI {


struct Label : Widget {
    static constexpr float BW = 8.f;

    Label() {
        bbox = {0, 0, 300, 40};
    }

    std::string text;

    float font_size = 20.f;
    FTGL::TextAlignment alignment = FTGL::ALIGN_LEFT;

    void paint() const override {
        glColor3f(0.25f, 0.25f, 0.25f);
        glRectf(bbox.x0, bbox.y0, bbox.x0 + bbox.nx, bbox.y0 + bbox.ny);

        Font font("assets/regular.ttf");
        font.set_font_size(font_size);
        font.set_fixed_width(bbox.nx - BW * 2, alignment);
        font.set_fixed_height(bbox.ny);
        glColor3f(1.f, 1.f, 1.f);

        font.render(bbox.x0 + BW, bbox.y0, text);
    }
};


}  // namespace zeno::UI
