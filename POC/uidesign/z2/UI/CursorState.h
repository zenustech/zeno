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
    float tx = 0, ty = 0, s = 1;

    std::vector<Event> events;
    Widget *focus_widget = nullptr;
    bool need_repaint = true;

    void on_update();
    void after_update();
    void init_callbacks();
    void focus_on(Widget *widget);
    ztd::dtor_function translate(float dx, float dy, float ds = 1.f);
    static AABB update_transforms();
    void update_window(Widget *win);
    bool is_invalid();
};

extern CursorState cur;


}  // namespace z2::UI
