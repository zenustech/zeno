#include <cstdio>
#include <cstdlib>
#include <GL/gl.h>
#include <GLFW/glfw3.h>

GLFWwindow *window;

void process_input() {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}

int main() {
    if (!glfwInit()) {
        const char *err = "Unknown"; glfwGetError(&err);
        printf("Failed to initialize GLFW library: %s\n", err);
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    //glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
    window = glfwCreateWindow(800, 600, "Zeno Editor", nullptr, nullptr);
    if (!window) {
        const char *err = "Unknown"; glfwGetError(&err);
        printf("Failed to create GLFW window: %s\n", err);
        return -1;
    }
    glfwMakeContextCurrent(window);

    while (!glfwWindowShouldClose(window)) {
        process_input();

        glViewport(0, 0, 800, 600);
        glClearColor(0.2f, 0.3f, 0.5f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glRectf(0.f, 0.f, 1.f, 1.f);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    return 0;
}
