#include <zeno/zeno.h>
#include <zeno/types/FunctionObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/zlog.h>
#include <GLES2/gl2.h>
#include <GL/glut.h>

struct GLUTMainLoop : zeno::INode {
    inline static std::shared_ptr<zeno::FunctionObject> drawFunc;
    inline static int interval;
    inline static int nx, ny;

    static void displayFunc() {
        glViewport(0, 0, nx, ny);
        glClearColor(0.23f, 0.23f, 0.23f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        zlog::trace("calling draw function...");
        drawFunc->call({});
        glFlush();
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
        zlog::trace("entering main loop...");
        glutMainLoop();
    }

    virtual void apply() override {
        drawFunc = get_input<zeno::FunctionObject>("drawFunc");
        zeno::vec2i resolution(512, 512);
        if (has_input("resolution"))
            resolution = get_input<zeno::NumericObject>(
                    "resolution")->get<zeno::vec2i>();
        nx = resolution[0];
        ny = resolution[1];
        interval = get_param<int>("interval");
        zlog::debug("initializing with res={}x{} itv={}", nx, ny, interval);
        mainFunc();
    }
};

ZENDEFNODE(GLUTMainLoop, {
        {"drawFunc", "resolution"},
        {},
        {{"int", "interval", "17 0"}},
        {"EasyGL"},
});
