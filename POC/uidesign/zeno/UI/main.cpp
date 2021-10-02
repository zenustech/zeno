#include <zeno/UI/stdafx.h>
#include <zeno/UI/CursorState.h>
#include <zeno/UI/UiDopGraph.h>
#include <zeno/UI/UiDopEditor.h>
#if defined(__linux__)
#include <unistd.h>
#endif


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
    if (glfwGetKey(cur.window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(cur.window, GLFW_TRUE);
    }
    cur.update_window(&win);
}


void draw_graphics() {
    if (cur.is_invalidated()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        win.do_paint();
        glfwSwapBuffers(cur.window);
    }
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
    cur.init_callbacks();

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
