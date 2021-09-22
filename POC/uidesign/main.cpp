#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <GL/gl.h>
#include <GLFW/glfw3.h>
#include <tuple>

GLFWwindow *window;


std::tuple<float, float> get_cursor_pos() {
    double x, y;
    glfwGetCursorPos(window, &x, &y);
    return {(float)x, (float)y};
}


struct Button {
    float x0, y0, nx, ny;

    bool hover = false;

    void on_event() {
        auto [x, y] = get_cursor_pos();
        printf("%f %f\n", x, y);

        hover = (x0 <= x && y0 <= y && x <= x0 + nx && y <= y0 + ny);
    }

    void on_draw() const {
        if (hover) {
            glColor3f(0.375f, 0.5f, 1.0f);
        } else {
            glColor3f(0.375f, 0.375f, 0.375f);
        }
        glRectf(x0, y0, x0 + nx, y0 + ny);
    }
};


Button btn{100, 100, 400, 400};


void process_input() {
    GLint nx = 100, ny = 100;
    glfwGetFramebufferSize(window, &nx, &ny);
    glViewport(0, 0, nx, ny);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glScalef(2.f, -2.f, 1.f);
    glTranslatef(-.5f, -.5f, 1.f);
    glScalef(1.f / nx, 1.f / ny, 1.f);

    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }

    btn.on_event();
}


void draw_graphics() {
    glClearColor(0.2f, 0.3f, 0.5f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    btn.on_draw();
}


int main() {
    if (!glfwInit()) {
        const char *err = "Unknown"; glfwGetError(&err);
        printf("Failed to initialize GLFW library: %s\n", err);
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    window = glfwCreateWindow(800, 600, "Zeno Editor", nullptr, nullptr);
    if (!window) {
        const char *err = "Unknown"; glfwGetError(&err);
        printf("Failed to create GLFW window: %s\n", err);
        return -1;
    }
    glfwMakeContextCurrent(window);

    while (!glfwWindowShouldClose(window)) {
        process_input();
        draw_graphics();
        glfwSwapBuffers(window);
        glfwPollEvents();
        usleep(16000);
    }

    return 0;
}
