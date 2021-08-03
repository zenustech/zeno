#include "GLShaderObject.h"
#include "GLProgramObject.h"
#include "GLVertexAttribInfo.h"
#include "GLTextureObject.h"
#include "GLFramebuffer.h"
#include <GL/glut.h>

inline static int interval = 100;
inline static int nx = 512, ny = 512;

static float triangle[] = {
    0.0f, 0.5f,
    -0.5f, -0.5f,
    0.5f, -0.5f,
};

static float rectangle[] = {
    0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1,
};

static GLProgramObject prog1, prog2;
static GLTextureFramebuffer fbo;

static void initFunc() {
    GLShaderObject vert, frag;
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
    prog1.initialize({vert, frag});
    vert.initialize(GL_VERTEX_SHADER, R"(
#version 300 es
layout (location = 0) in vec3 vPosition;
out vec2 fragCoord;
void main() {
  fragCoord = vPosition.xy;
  gl_Position = vec4(vPosition * 2.0 - 1.0, 1.0);
}
)");
    frag.initialize(GL_FRAGMENT_SHADER, R"(
#version 300 es
precision mediump float;
uniform sampler2D image;
in vec2 fragCoord;
out vec4 fragColor;
vec3 at(int dx, int dy) {
  vec2 uv = fragCoord + vec2(dx, dy) / 512.0;
  vec3 color = texture(image, uv).xyz;
  return color;
}
void main() {
  vec3 color = vec3(0);
  for (int i = -10; i <= 10; i++) {
    color += at(i, 0);
  }
  color /= 21.0;
  fragColor = vec4(color, 1.0);
}
)");
    prog2.initialize({vert, frag});

    fbo.colorTextures.resize(1);
    fbo.colorTextures[0].width = 512;
    fbo.colorTextures[0].height = 512;
    fbo.colorTextures[0].initialize();
    fbo.initialize();
}

static void drawFunc() {
    prog1.use();
    fbo.use();
    glClearColor(0.23f, 0.23f, 0.23f, 0.f);
    glClear(GL_COLOR_BUFFER_BIT);
    GLVertexAttribInfo vab;
    vab.base = triangle;
    vab.dim = 2;
    drawVertexArrays(GL_TRIANGLES, 3, {vab});

    prog2.use();
    GLFramebuffer().use();
    vab.base = rectangle;
    vab.dim = 2;
    fbo.colorTextures[0].use(0);
    drawVertexArrays(GL_TRIANGLES, 6, {vab});
}

static void displayFunc() {
    glViewport(0, 0, nx, ny);
    spdlog::info("calling draw function...");
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
