#include "CursorState.h"


CursorState cur;


void CursorState::on_update() {
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

void CursorState::after_update() {
    if (events.size() || dx || dy)
        need_repaint = true;
    events.clear();
}
