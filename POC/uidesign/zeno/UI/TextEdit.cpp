#include <zeno/UI/TextEdit.h>


namespace zeno::UI {


void TextEdit::on_event(Event_Hover e) {
    Widget::on_event(e);

    if (e.enter == false && !disabled)
        on_editing_finished();
}

void TextEdit::on_event(Event_Mouse e) {
    Widget::on_event(e);

    if (disabled)
        return;
    if (e.down != true)
        return;
    if (e.btn != 0)
        return;

    if (cursor == 0 && sellen == text.size()) {
        cursor = text.size();
        sellen = 0;
    } else {
        cursor = 0;
        sellen = text.size();
    }
}

void TextEdit::on_event(Event_Key e) {
    Widget::on_event(e);

    if (e.down != true)
        return;

    if (e.key == GLFW_KEY_C && e.mode == GLFW_MOD_CONTROL) {
        auto str = sellen ? text.substr(cursor, sellen) : text;
        if (str.size())
            glfwSetClipboardString(cur.window, str.c_str());

    } else if (e.key == GLFW_KEY_A && e.mode == GLFW_MOD_CONTROL) {
        cursor = 0;
        sellen = text.size();
    }

    if (disabled) {

    } else if (e.key == GLFW_KEY_V && e.mode == GLFW_MOD_CONTROL) {
        if (auto str = glfwGetClipboardString(cur.window); str)
            _insert_text(str);

    } else if (e.key == GLFW_KEY_LEFT) {
        cursor = std::max(0, cursor - 1);
        sellen = 0;

    } else if (e.key == GLFW_KEY_RIGHT) {
        cursor = std::min((int)text.size(), cursor + 1 + sellen);
        sellen = 0;

    } else if (e.key == GLFW_KEY_BACKSPACE) {
        if (sellen) {
            _insert_text("");
        } else if (cursor - 1 > 0) {
            text = text.substr(0, cursor - 1) + text.substr(cursor);
            cursor = std::max(0, cursor - 1);
        } else {
            text = text.substr(cursor);
            cursor = std::max(0, cursor - 1);
        }

    }
}

void TextEdit::on_event(Event_Char e) {
    Widget::on_event(e);

    if (disabled)
        return;
    char c = e.code;
    _insert_text(ztd::to_string(c));
}

void TextEdit::paint() const {
    if (disabled) {
        glColor3f(0.275f, 0.275f, 0.275f);
    } else if (hovered) {
        glColor3f(0.375f, 0.5f, 1.0f);
    } else {
        glColor3f(0.375f, 0.375f, 0.375f);
    }
    glRectf(bbox.x0, bbox.y0, bbox.x0 + bbox.nx, bbox.y0 + bbox.ny);

    Font font("assets/regular.ttf");
    font.set_font_size(font_size);
    font.set_fixed_width(bbox.nx - BW * 2, alignment);
    font.set_fixed_height(bbox.ny);
    if (disabled) {
        glColor3f(0.65f, 0.65f, 0.65f);
    } else {
        glColor3f(1.f, 1.f, 1.f);
    }

    auto txt = !hovered || disabled ? text : sellen == 0
        ? text.substr(0, cursor) + '|' + text.substr(cursor)
        : text.substr(0, cursor) + '|' + text.substr(cursor, sellen) + '|' + text.substr(cursor + sellen);
    font.render(bbox.x0 + BW, bbox.y0, txt);
}


}  // namespace zeno::UI
