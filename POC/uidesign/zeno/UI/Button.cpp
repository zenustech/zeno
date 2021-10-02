#include <zeno/UI/Button.h>


void Button::on_event(Event_Mouse e) {
    Widget::on_event(e);

    if (e.down != false)
        return;
    if (e.btn != 0)
        return;

    on_clicked();
}


void Button::paint() const {
    if (pressed[0]) {
        glColor3f(0.75f, 0.5f, 0.375f);
    } else if (hovered) {
        glColor3f(0.375f, 0.5f, 1.0f);
    } else {
        glColor3f(0.375f, 0.375f, 0.375f);
    }
    glRectf(bbox.x0, bbox.y0, bbox.x0 + bbox.nx, bbox.y0 + bbox.ny);

    if (text.size()) {
        Font font("assets/regular.ttf");
        font.set_font_size(font_size);
        font.set_fixed_width(bbox.nx - BW * 2, alignment);
        font.set_fixed_height(bbox.ny);
        glColor3f(1.f, 1.f, 1.f);
        font.render(bbox.x0 + BW, bbox.y0, text);
    }
}
