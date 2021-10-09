#include <zeno2/UI/Font.h>


namespace zeno2::UI {


Font get_default_font() {
    static const char data[] =
#include <zeno2/assets/regular.ttf.h>
    return Font(data, size);
}


}