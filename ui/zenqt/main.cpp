//#include <Python.h>
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
    ZenoApplication a(argc, argv);
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
    return a.exec();
}
