#pragma once


#include "Event.h"


struct Widget;

struct CursorState {
    float x = 0, y = 0;
    float dx = 0, dy = 0;
    float last_x = 0, last_y = 0;
    bool lmb = false, mmb = false, rmb = false;
    bool shift = false, ctrl = false, alt = false;

    void on_update() {
        last_x = x;
        last_y = y;

        GLint nx, ny;
        glfwGetFramebufferSize(window, &nx, &ny);
        GLdouble _x, _y;
        glfwGetCursorPos(window, &_x, &_y);
        x = 0.5f + (float)_x;
        y = ny - 0.5f - (float)_y;
        dx = x - last_x;
        dy = y - last_y;
        lmb = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        mmb = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
        rmb = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
        shift = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS;
        ctrl = glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS;
        alt = glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS;
    }

    bool is_pressed(int key) const {
        return glfwGetKey(window, key) == GLFW_PRESS;
    }

    auto translate(float dx, float dy) {
        auto ox = x, oy = y;
        x += dx; y += dy;
        struct RAII : std::function<void()> {
            using std::function<void()>::function;
            ~RAII() { (*this)(); }
        } raii {[=, this] () {
            x = ox; y = oy;
        }};
        return raii;
    }

    std::vector<Event> events;

    bool need_repaint = true;

    void after_update() {
        if (events.size() || dx || dy)
            need_repaint = true;
        events.clear();
    }
};

CursorState cur;


static void cursor_pos_callback(GLFWwindow *window, double xpos, double ypos) {
    //GLint nx, ny;
    //glfwGetFramebufferSize(window, &nx, &ny);
    //auto x = 0.5f + (float)xpos;
    //auto y = ny - 0.5f - (float)ypos;
    //cur.events.push_back(Event_Motion{.x = x, .y = y});
}

static void mouse_button_callback(GLFWwindow *window, int btn, int action, int mode) {
    cur.events.push_back(Event_Mouse{.btn = btn, .down = action == GLFW_PRESS});
}

static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mode) {
    cur.events.push_back(Event_Key{.key = key, .mode = mode, .down = action == GLFW_PRESS});
}

static void char_callback(GLFWwindow *window, unsigned int codeprint) {
    cur.events.push_back(Event_Char{.code = codeprint});
}


