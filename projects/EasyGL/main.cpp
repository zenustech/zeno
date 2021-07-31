#include <zeno/zeno.h>
#include <GLES2/gl2.h>
#include <GL/glut.h>


static std::unique_ptr<zeno::Scene> scene;

void drawScene() {
    scene->getGraph().applyGraph();
}

void initFunc() {
    scene = zeno::createScene();
    auto json = (
#include "myprogram.inl.h"
            );
    scene->loadScene(json);
}

void displayFunc() {
    glClearColor(0.375f, 0.75f, 1.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    drawScene();
    glFlush();
}

#define ITV 100
void timerFunc(int unused) {
    glutPostRedisplay();
    glutTimerFunc(ITV, timerFunc, 0);
}

void keyboardFunc(unsigned char key, int x, int y) {
    if (key == 27)
        exit(0);
}

int main(int argc, char **argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_SINGLE | GLUT_RGBA);
    glutInitWindowPosition(100, 100);
    glutInitWindowSize(512, 512);
    glutCreateWindow("GLUT Window");
    glutDisplayFunc(displayFunc);
    glutKeyboardFunc(keyboardFunc);
    initFunc();
    glutTimerFunc(ITV, timerFunc, 0);
    glutMainLoop();
}
