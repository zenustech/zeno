#include <Python.h>
#include <QApplication>
#include "style/zenostyle.h"
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include "startup/zstartup.h"
#include "settings/zsettings.h"
#include "zeno/utils/log.h"
#include "zeno/zeno.h"
#include "zeno/extra/EventCallbacks.h"
#include "startup/pythonenv.h"
#include <zeno/utils/Error.h>


/* debug cutsom layout: ZGraphicsLayout */
//#define DEBUG_ZENOGV_LAYOUT
//#define DEBUG_NORMAL_WIDGET

#ifdef DEBUG_TESTPYBIND
PyMODINIT_FUNC PyInit_spam(void);
#endif

#ifdef DEBUG_ZENOGV_LAYOUT
#include <zenoui/comctrl/gv/gvtestwidget.h>
#endif

#ifdef DEBUG_NORMAL_WIDGET
#include <zenoui/comctrl/testwidget.h>
#endif


int main(int argc, char *argv[]) 
{
    //Py_Initialize();

    //// 获取主模块的字典
    //PyObject* main_module = PyImport_ImportModule("__main__");
    //if (main_module == NULL) {
    //    printf("Failed to import __main__!\n");
    //    PyErr_Print();
    //    Py_Finalize();
    //    return 1;
    //}
    //PyObject* globals = PyModule_GetDict(main_module);
    //Py_DECREF(main_module);

    //// 编译import pandas语句
    //PyObject* code = Py_CompileString("import pandas", "<string>", Py_file_input);
    //if (code == NULL) {
    //    printf("Failed to compile 'import pandas'!\n");
    //    PyErr_Print();
    //    Py_Finalize();
    //    return 1;
    //}

    //// 执行编译后的代码，传入全局和局部命名空间
    //PyObject* result = PyEval_EvalCode(code, globals, globals);
    //Py_DECREF(code);

    //if (result == NULL) {
    //    printf("Failed to execute 'import pandas'!\n");
    //    PyErr_Print();
    //    Py_Finalize();
    //    return 1;
    //}
    //Py_DECREF(result);

    //// 从全局命名空间中获取pandas模块（import语句会将模块添加到globals中）
    //PyObject* pandas_module = PyDict_GetItemString(globals, "pandas");
    //if (pandas_module == NULL) {
    //    printf("Failed to get pandas from globals!\n");
    //    PyErr_Print();
    //    Py_Finalize();
    //    return 1;
    //}

    //// 获取版本
    //PyObject* version = PyObject_GetAttrString(pandas_module, "__version__");
    //if (version == NULL) {
    //    printf("Failed to get pandas version!\n");
    //    PyErr_Print();
    //    Py_Finalize();
    //    return 1;
    //}

    //const char* ver = PyUnicode_AsUTF8(version);
    //printf("Successfully imported pandas!\n");
    //printf("Pandas version: %s\n", ver);

    //Py_DECREF(version);
    //Py_Finalize();

    ZenoApplication a(argc, argv);

    qRegisterMetaType<std::vector<std::shared_ptr<zeno::IObject>>>();

    a.setStyle(new ZenoStyle);

#ifdef DEBUG_NORMAL_WIDGET
    TestNormalWidget wid;
    wid.show();
    return a.exec();
#endif

#ifdef DEBUG_ZENOGV_LAYOUT
    TestGraphicsView view;
    view.show();
    return a.exec();
#endif

#ifdef ZENO_WITH_PYTHON3
    initPythonEnv(argv[0]);
#endif

    if (argc >= 3 && !strcmp(argv[1], "--optixcmd")) {
        extern int optixcmd(const QCoreApplication & app, int port);
        int port = atoi(argv[2]);
        startUp(false);
        return optixcmd(a, port);
    }

    //entrance for the zenoedit-player.
    if (argc >= 2 && !strcmp(argv[1], "--record"))
    {
        extern int record_main(const QCoreApplication & app);
        startUp(false);
        return record_main(a);
    }

    if (argc >= 3 && !strcmp(argv[1], "--offline")) {
        extern int offline_main(const QCoreApplication & app);
        startUp(false);
        return offline_main(a);
    }
    if (argc >= 3 && !strcmp(argv[1], "--blender"))
    {
        extern int blender_main(const QCoreApplication & app);
        startUp(false);
        return blender_main(a);
    }
    if (argc >= 2 && !strcmp(argv[1], "--cmdtool"))
    {
        extern int main_cmdtool(const ZenoApplication & app, int argc);
        startUp(false);
        return main_cmdtool(a, argc);
    }

#ifdef ZENO_MULTIPROCESS
    if (argc >= 2 && !strcmp(argv[1], "--runner")) {
        extern int runner_main(const QCoreApplication & app);
        startUp(false);
        return runner_main(a);
    }

    startUp(true);

    if (argc >= 3 && !strcmp(argv[1], "-optix")) {
        //MessageBox(0, "runner", "runner", MB_OK);
        extern int optix_main(const QCoreApplication & app, 
                            int port,
                            const char* cachedir,
                            int cachenum,
                            int sFrame,
                            int eFrame,
                            int finishedFrames,
                            const char* sessionId);
        int port = -1;
        if (argc >= 5 && !strcmp(argv[3], "-port"))
            port = atoi(argv[4]);
        char* cachedir = nullptr;
        int cachenum = 0, sFrame = 0, eFrame = 0;
        int finishedFrames = 0;
        char* sessionId = nullptr;
        if (argc >= 7 && !strcmp(argv[5], "-cachedir"))
            cachedir = argv[6];
        if (argc >= 9 && !strcmp(argv[7], "-cachenum"))
            cachenum = atoi(argv[8]);
        if (argc >= 11 && !strcmp(argv[9], "-beginFrame"))
            sFrame = atoi(argv[10]);
        if (argc >= 13 && !strcmp(argv[11], "-endFrame"))
            eFrame = atoi(argv[12]);
        if (argc >= 15 && !strcmp(argv[13], "-finishedFrames"))
            finishedFrames = atoi(argv[14]);
        if (argc >= 17 && !strcmp(argv[15], "-sessionId"))
            sessionId = argv[16];
        return optix_main(a, port, cachedir, cachenum, sFrame, eFrame, finishedFrames, sessionId);
    }
#endif

    if (argc >= 3 && !strcmp(argv[1], "-invoke")) {
        extern int invoke_main(int argc, char *argv[]);
        return invoke_main(argc - 2, argv + 2);
    }

    QTranslator t;
    QTranslator qtTran;
    QSettings settings(zsCompanyName, zsEditor);
    QVariant use_chinese = settings.value("use_chinese");

    if (use_chinese.isNull() || use_chinese.toBool()) {
        if (t.load(":languages/zh.qm")) {
            a.installTranslator(&t);
        }
        if (qtTran.load(":languages/qt_zh_CN.qm")) {
            a.installTranslator(&qtTran);
        }
    }

    ZenoMainWindow mainWindow;
    zeno::getSession().eventCallbacks->triggerEvent("editorConstructed");
    mainWindow.showMaximized();
    if (argc >= 2) {
        QCommandLineParser cmdParser;
        cmdParser.addHelpOption();
        cmdParser.addOptions({
            {"zsg", "zsg", "zsg"},
            {"paramsJson", "paramsJson", "paramsJson"}
        });
        cmdParser.process(a);
        QString zsgPath;
        if (cmdParser.isSet("zsg"))
            zsgPath = cmdParser.value("zsg");
        QString paramsJson;
        if (cmdParser.isSet("paramsJson"))
            paramsJson = cmdParser.value("paramsJson");
        if (!zsgPath.isEmpty())
            mainWindow.openFileAndUpdateParam(zsgPath, paramsJson);
    }

    return a.exec();
}
