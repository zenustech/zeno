#include <zs/editor/UI/Font.h>


namespace zs::editor::UI {


Font get_default_font() {
    static const uint8_t data[] = {
#include <zs/zeno/assets/regular.ttf.inl>
    };
    return Font(data, sizeof(data));
}


}