#pragma once


#include <zeno2/UI/AABB.h>


namespace zeno2::UI {


struct Font {
    std::unique_ptr<FTFont> font;
    std::unique_ptr<FTSimpleLayout> layout;
    float fixed_height = -1;

    Font(const char *path);
    Font &set_font_size(float font_size);
    Font &set_fixed_width(float width, FTGL::TextAlignment align = FTGL::ALIGN_CENTER);
    Font &set_fixed_height(float height);
    AABB calc_bounding_box(std::string const &str);
    Font &render(float x, float y, std::string const &str);
};


Font get_default_font();


}  // namespace zeno2::UI
