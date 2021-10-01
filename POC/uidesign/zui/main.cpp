#include "zui/stdafx.h"
#include "zui/CursorState.h"
#include "zui/UiDopGraph.h"
#include "zui/UiDopEditor.h"

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

struct RootWindow : Widget {
    UiDopGraph *graph;
    UiDopEditor *editor;

    RootWindow() {
        graph = add_child<UiDopGraph>();
        graph->bbox = {0, 0, 1024, 512};
        graph->position = {0, 256};
        editor = add_child<UiDopEditor>();
        editor->bbox = {0, 0, 1024, 256};
        graph->editor = editor;
    }
} win;

void process_input() {
    GLint nx = 100, ny = 100;
    glfwGetFramebufferSize(cur.window, &nx, &ny);
    glViewport(0, 0, nx, ny);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glScalef(2.f, 2.f, -.001f);
    glTranslatef(-.5f, -.5f, 1.f);
    glScalef(1.f / nx, 1.f / ny, 1.f);

    if (glfwGetKey(cur.window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(cur.window, GLFW_TRUE);
    }

    cur.on_update();
    win.bbox = {0, 0, (float)nx, (float)ny};
    win.do_update();
    win.do_update_event();
    win.after_update();
    cur.after_update();
}


void draw_graphics() {
    if (cur.need_repaint) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        win.do_paint();
        glfwSwapBuffers(cur.window);
        cur.need_repaint = false;
    }
}


static void window_refresh_callback(GLFWwindow *window) {
    cur.need_repaint = true;
}

int main() {
    if (!glfwInit()) {
        const char *err = "unknown error"; glfwGetError(&err);
        fprintf(stderr, "Failed to initialize GLFW library: %s\n", err);
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
    cur.window = glfwCreateWindow(1024, 768, "Zeno Editor", nullptr, nullptr);
    glfwSetWindowPos(cur.window, 0, 0);
    if (!cur.window) {
        const char *err = "unknown error"; glfwGetError(&err);
        fprintf(stderr, "Failed to create GLFW window: %s\n", err);
        return -1;
    }
    glfwMakeContextCurrent(cur.window);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        fprintf(stderr, "Failed to initialize GLAD\n");
        return -1;
    }

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    glEnable(GL_POINT_SMOOTH);
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
    glfwSetKeyCallback(cur.window, key_callback);
    glfwSetCharCallback(cur.window, char_callback);
    glfwSetMouseButtonCallback(cur.window, mouse_button_callback);
    glfwSetCursorPosCallback(cur.window, cursor_pos_callback);
    glfwSetWindowRefreshCallback(cur.window, window_refresh_callback);

    double lasttime = glfwGetTime();
    while (!glfwWindowShouldClose(cur.window)) {
        glfwWaitEvents();
        process_input();
        draw_graphics();
#if 0
        lasttime += 1.0 / fps;
        while (glfwGetTime() < lasttime) {
            double sleepfor = (lasttime - glfwGetTime()) * 0.75;
            int us(sleepfor / 1000000);
            std::this_thread::sleep_for(std::chrono::microseconds(us));
        }
#endif
    }

    return 0;
}

// END ui library main loop
