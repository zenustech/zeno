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

    void on_update();
    void after_update();

    bool is_pressed(int key) const {
        return glfwGetKey(window, key) == GLFW_PRESS;
    }

    ztd::CallOnDtor translate(float dx, float dy) {
        auto ox = x, oy = y;
        x += dx; y += dy;
        return [=, this] () {
            x = ox; y = oy;
        };
    }

    std::vector<Event> events;

    bool need_repaint = true;

    void init_callbacks();
    static AABB update_transforms();
    void update_window(Widget *win);

    bool is_invalidated() {
        if (need_repaint) {
            need_repaint = false;
            return true;
        } else {
            return false;
        }
    }
};

extern CursorState cur;


}  // namespace z2::UI
