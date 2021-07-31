#include "GLShaderObject.h"
#include "GLProgramObject.h"
#include <GL/glut.h>

inline static int interval = 100;
inline static int nx = 512, ny = 512;

static void displayFunc() {
    glViewport(0, 0, nx, ny);
    spdlog::info("calling draw function...");
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glFlush();
}

static void timerFunc(int interval) {
    glutPostRedisplay();
    glutTimerFunc(interval, timerFunc, interval);
}

static void keyboardFunc(unsigned char key, int, int) {
    if (key == 27)
        exit(0);
}

int main() {
    int argc = 1;
    const char *argv[] = {"make_glut_happy", NULL};
    glutInit(&argc, (char **)argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_SINGLE | GLUT_RGBA);
    glutInitWindowPosition(100, 100);
    glutInitWindowSize(nx, ny);
    glutCreateWindow("GLUT Window");
    glutDisplayFunc(displayFunc);
    glutKeyboardFunc(keyboardFunc);
    glutTimerFunc(interval, timerFunc, interval);
    spdlog::info("entering main loop...");
    glutMainLoop();
}
