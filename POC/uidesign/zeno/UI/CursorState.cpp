#include "CursorState.h"
#include "Widget.h"


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

static void window_refresh_callback(GLFWwindow *window) {
    cur.need_repaint = true;
}

void CursorState::init_callbacks() {
    glfwSetKeyCallback(window, key_callback);
    glfwSetCharCallback(window, char_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_pos_callback);
    glfwSetWindowRefreshCallback(window, window_refresh_callback);
}

AABB CursorState::update_transforms() {
    GLint nx = 100, ny = 100;
    glfwGetFramebufferSize(cur.window, &nx, &ny);
    glViewport(0, 0, nx, ny);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glScalef(2.f, 2.f, -.001f);
    glTranslatef(-.5f, -.5f, 1.f);
    glScalef(1.f / nx, 1.f / ny, 1.f);
    return {0, 0, (float)nx, (float)ny};
}

void CursorState::update_window(Widget *win) {
    auto bbox = update_transforms();
    on_update();
    win->bbox = bbox;
    win->do_update();
    win->do_update_event();
    win->after_update();
    after_update();
}
