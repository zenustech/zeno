#include <zenoio/reader/zsgreader.h>
#include "zenoapplication.h"
#include <zenomodel/include/graphsmanagment.h>
#include "zenomainwindow.h"
#include <zeno/utils/log.h>
#include "util/log.h"
#include "launch/ztcpserver.h"
#include "launch/corelaunch.h"
#include "startup/zstartup.h"


ZenoApplication::ZenoApplication(int &argc, char **argv)
    : QApplication(argc, argv)
    , m_errSteam(std::clog)
#if defined(ZENO_MULTIPROCESS) && defined(ZENO_IPC_USE_TCP)
    , m_server(nullptr)
#endif
{
    initFonts();
    initStyleSheets();
    m_errSteam.registerMsgHandler();
    verifyVersion();

    QStringList locations;
    locations = QStandardPaths::standardLocations(QStandardPaths::AppDataLocation);
#ifdef Q_OS_WIN
    locations = locations.filter("ProgramData");
    ZASSERT_EXIT(!locations.isEmpty());
    m_appDataPath.setPath(locations[0]);
#endif
}

ZenoApplication::~ZenoApplication()
{
}

QString ZenoApplication::readQss(const QString& qssPath)
{
    bool ret = false;
    QFile file;
    file.setFileName(qssPath);
    ret = file.open(QIODevice::ReadOnly | QIODevice::Text);
    ZASSERT_EXIT(ret, "");
    return file.readAll();
}

void ZenoApplication::initStyleSheets()
{
    QString qss;
    qss += readQss(":/stylesheet/qlabel.qss");
    qss += readQss(":/stylesheet/qlineedit.qss");
    qss += readQss(":/stylesheet/menu.qss");
    qss += readQss(":/stylesheet/qcombobox.qss");
    qss += readQss(":/stylesheet/qwidget.qss");
    qss += readQss(":/stylesheet/pushbutton.qss");
    qss += readQss(":/stylesheet/scrollbar.qss");
    qss += readQss(":/stylesheet/spinbox.qss");
    qss += readQss(":/stylesheet/mainwindow.qss");
    setStyleSheet(qss);
}

void ZenoApplication::initFonts()
{
    QFontDatabase::addApplicationFont(":/font/HarmonyOS_Sans/HarmonyOS_Sans_Black.ttf");
    QFontDatabase::addApplicationFont(":/font/HarmonyOS_Sans/HarmonyOS_Sans_Regular.ttf");
    QFontDatabase::addApplicationFont(":/font/HarmonyOS_Sans/HarmonyOS_Sans_Light.ttf");
    QFontDatabase::addApplicationFont(":/font/HarmonyOS_Sans/HarmonyOS_Sans_Medium.ttf");
    QFontDatabase::addApplicationFont(":/font/HarmonyOS_Sans/HarmonyOS_Sans_Thin.ttf");
    QFontDatabase::addApplicationFont(":/font/HarmonyOS_Sans/HarmonyOS_Sans_Bold.ttf");

    //QFontDatabase::addApplicationFont(":/font/HarmonyOS_Sans_SC/HarmonyOS_Sans_SC_Black.ttf");
    //QFontDatabase::addApplicationFont(":/font/HarmonyOS_Sans_SC/HarmonyOS_Sans_SC_Bold.ttf");
    //QFontDatabase::addApplicationFont(":/font/HarmonyOS_Sans_SC/HarmonyOS_Sans_SC_Light.ttf");
    //QFontDatabase::addApplicationFont(":/font/HarmonyOS_Sans_SC/HarmonyOS_Sans_SC_Medium.ttf");
    //QFontDatabase::addApplicationFont(":/font/HarmonyOS_Sans_SC/HarmonyOS_Sans_SC_Regular.ttf");
    //QFontDatabase::addApplicationFont(":/font/HarmonyOS_Sans_SC/HarmonyOS_Sans_SC_Thin.ttf");
}

GraphsManagment *ZenoApplication::graphsManagment() const
{
    return &GraphsManagment::instance();
}

QStandardItemModel* ZenoApplication::logModel() const
{
    return graphsManagment()->logModel();
}

#if defined(ZENO_MULTIPROCESS) && defined(ZENO_IPC_USE_TCP)
ZTcpServer* ZenoApplication::getServer()
{
    if (!m_server) {
        m_server = new ZTcpServer(this);
        m_server->init(QHostAddress::LocalHost);
    }
    return m_server;
}
#endif

ZenoMainWindow* ZenoApplication::getMainWindow()
{
	foreach(QWidget* widget, topLevelWidgets())
		if (ZenoMainWindow* mainWindow = qobject_cast<ZenoMainWindow*>(widget))
			return mainWindow;
    return nullptr;
}

QWidget *ZenoApplication::getWindow(const QString &objName)
{
    foreach (QWidget *widget, QApplication::allWidgets())
        if(widget->objectName() == objName)
            return widget;
    return nullptr;
}
