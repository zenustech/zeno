#include <zeno2/UI/Font.h>


namespace zeno2::UI {


Font get_default_font() {
    static const uint8_t data[] = {
#include <zeno2/assets/regular.ttf.inl>
    };
    return Font(data, sizeof(data));
}


}