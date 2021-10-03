#include <z2/UI/CursorState.h>
#include <z2/UI/UiDopScene.h>
#include <z2/UI/UiVisViewport.h>
#if defined(__linux__)
#include <unistd.h>
#endif


namespace z2::UI {


struct UiMainWindow : Widget {
    UiDopScene *scene;
    UiVisViewport *viewport;

    UiMainWindow() {
        viewport = add_child<UiVisViewport>();
        scene = add_child<UiDopScene>();
    }
};


std::unique_ptr<Widget> win;


static void process_input() {
    if (glfwGetKey(cur.window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(cur.window, GLFW_TRUE);
    }
    cur.update_window(win.get());
}


static void draw_graphics() {
    if (cur.is_invalid()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        win->do_paint();
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
    win = std::make_unique<UiMainWindow>();

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


}  // namespace z2::UI


int main() {
    return z2::UI::main();
}
