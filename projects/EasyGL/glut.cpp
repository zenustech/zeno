#include <GL/freeglut_std.h>
#include <zeno/zeno.h>
#include <zeno/types/FunctionObject.h>
#include <zeno/types/NumericObject.h>
//#include <zeno/utils/zlog.h>
#include <GLES2/gl2.h>
#include <GL/glut.h>
#include <zeno/types/DictObject.h>
static float lastx = 0, lasty = 0;
static float fx=0,fy=0,ax=0,ay=0;
static bool clicked = false;
struct GLUTMainLoop : zeno::INode {
    
    inline static std::shared_ptr<zeno::FunctionObject> drawFunc;
    inline static int interval;
    inline static int nx, ny;

    static void click(int button, int updown, int x, int y)
    {
        lastx = x;
        lasty = y;
        clicked = !clicked;
    }

    static void motion(int x, int y)
    {
        

        if (clicked)
        {
            int ddx = x - lastx;
            int ddy = y - lasty;
            fx = ddx;
            fy = -ddy;
            ax = x/(float)nx;
            ay = 1-y/(float)ny;
            lastx = x;
            lasty = y;
        }
        else {
            fx = 0;
            fy = 0;
        }

        glutPostRedisplay();
    }
    static void displayFunc() {
        glViewport(0, 0, nx, ny);
        glClearColor(0.23f, 0.23f, 0.23f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        //zlog::trace("calling draw function...");

        auto vec = zeno::vec4f(ax, ay, fx, fy);
        auto vecobj = std::make_shared<zeno::NumericObject>(vec);
        std::map<std::string, zeno::zany> param;
        param["mouseInput"] = vecobj;
        drawFunc->call(param);
        glutSwapBuffers();
        glutPostRedisplay();
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
        glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
        glutInitWindowPosition(100, 100);
        glutInitWindowSize(nx, ny);
        glutCreateWindow("GLUT Window");
        glutDisplayFunc(displayFunc);
        glutKeyboardFunc(keyboardFunc);
        glutMouseFunc(click);
        glutMotionFunc(motion);
        glutTimerFunc(interval, timerFunc, interval);
        //zlog::trace("entering main loop...");
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
        //zlog::debug("initializing with res={}x{} itv={}", nx, ny, interval);
        mainFunc();
    }
};

ZENDEFNODE(GLUTMainLoop, {
        {"drawFunc", "resolution"},
        {},
        {{"int", "interval", "17 0"}},
        {"EasyGL"},
});
