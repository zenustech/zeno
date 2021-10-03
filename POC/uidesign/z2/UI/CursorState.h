#pragma once


#include <z2/UI/Event.h>
#include <z2/UI/AABB.h>
#include <z2/ztd/functional.h>


namespace z2::UI {


struct Widget;

struct CursorState {
    GLFWwindow *window = nullptr;

    float x = 0, y = 0;
    float dx = 0, dy = 0;
    float last_x = 0, last_y = 0;
    bool lmb = false, mmb = false, rmb = false;
    bool shift = false, ctrl = false, alt = false;
    float tx = 0, ty = 0;

    void on_update();
    void after_update();

    std::vector<Event> events;

    bool need_repaint = true;

    void init_callbacks();
    ztd::dtor_function translate(float dx, float dy);
    static AABB update_transforms();
    void update_window(Widget *win);
    void update_cursor_pos();
    bool is_invalid();
};

extern CursorState cur;


}  // namespace z2::UI
