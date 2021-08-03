#include <zeno/zeno.h>
#include <zeno/types/FunctionObject.h>
#include <zeno/types/NumericObject.h>
#include <GLES2/gl2.h>
#include <GL/glut.h>

struct GLUTMainLoop : zeno::INode {
    inline static std::shared_ptr<zeno::FunctionObject> drawFunc;
    inline static int interval;
    inline static int nx, ny;

    static void displayFunc() {
        glViewport(0, 0, nx, ny);
        printf("calling draw function...\n");
        drawFunc->call({});
        glutPostRedisplay();
        //glFlush();
    }

    static void timerFunc(int interval) {
        glutPostRedisplay();
        glutTimerFunc(interval, timerFunc, interval);
    }

    static void keyboardFunc(unsigned char key, int x, int y) {
        if (key == 27)
            exit(0);
    }

    static void mainFunc() {
        const char *argv[] = {"make_glut_happy", NULL};
        int argc = sizeof(argv) / sizeof(argv[0]) - 1;
        glutInit(&argc, (char **)argv);
        glutInitDisplayMode(GLUT_DEPTH | GLUT_SINGLE | GLUT_RGBA);
        glutInitWindowPosition(100, 100);
        glutInitWindowSize(nx, ny);
        glutCreateWindow("GLUT Window");
        glutDisplayFunc(displayFunc);
        glutKeyboardFunc(keyboardFunc);
        glutTimerFunc(interval, timerFunc, interval);
        printf("entering main loop...\n");
        glutMainLoop();
    }

    virtual void apply() override {
        drawFunc = get_input<zeno::FunctionObject>("drawFunc");
        interval = get_param<int>("interval");
        nx = get_param<int>("nx");
        ny = get_param<int>("ny");
        mainFunc();
    }
};

ZENDEFNODE(GLUTMainLoop, {
        {"drawFunc"},
        {},
        {
            {"int", "interval", "17 0"},
            {"int", "nx", "512 1"},
            {"int", "ny", "512 1"},
        },
        {"EasyGL"},
});
