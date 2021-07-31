#include "GLShaderObject.h"
#include "GLProgramObject.h"
#include "GLVertexAttribInfo.h"
#include "GLTextureObject.h"
#include "GLFramebuffer.h"
#include <GL/glut.h>

inline static int interval = 100;
inline static int nx = 512, ny = 512;

static float vertices[] = {
    0.0f, 0.5f,
    -0.5f, -0.5f,
    0.5f, -0.5f,
};

static GLProgramObject prog;
static GLShaderObject vert, frag;
static GLFramebuffer fbo;
static GLTextureObject tex;

static void initFunc() {
    vert.initialize(GL_VERTEX_SHADER, R"(
#version 300 es
layout (location = 0) in vec2 vPosition;
void main() {
  gl_Position = vec4(vPosition, 0.0, 1.0);
}
)");
    frag.initialize(GL_FRAGMENT_SHADER, R"(
#version 300 es
precision mediump float;
out vec4 fragColor;
void main() {
  vec3 color = vec3(0.375, 0.75, 1.0);
  fragColor = vec4(color, 1.0);
}
)");
    prog.initialize({vert, frag});
}

static void drawFunc() {
    GLVertexAttribInfo vab;
    vab.base = vertices;
    vab.dim = 2;
    glUseProgram(prog);
    drawVertexArrays(GL_TRIANGLES, 3, {vab});
}

static void displayFunc() {
    glViewport(0, 0, nx, ny);
    spdlog::info("calling draw function...");
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    drawFunc();
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
    initFunc();
    glutMainLoop();
}
