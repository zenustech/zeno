#include <zenoio/reader/zsgreader.h>
#include "zenoapplication.h"
#include <zenomodel/include/graphsmanagment.h>
#include "zenomainwindow.h"
#include <zeno/utils/log.h>
#include "util/log.h"
#include "launch/ztcpserver.h"
#include "launch/corelaunch.h"
#include "startup/zstartup.h"
#include <style/zenostyle.h>
#include "settings/zenosettingsmanager.h"


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
    return ZenoStyle::dpiScaleSheet(file.readAll());
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
    qss += readQss(":/stylesheet/checkbox.qss");
    qss += readQss(":/stylesheet/tabwidget.qss");
    setStyleSheet(qss);
}

void ZenoApplication::initFonts()
{
    QFontDatabase::addApplicationFont(":/font/Noto_Sans_SC/NotoSansSC-Black.otf");
    QFontDatabase::addApplicationFont(":/font/Noto_Sans_SC/NotoSansSC-Bold.otf");
    QFontDatabase::addApplicationFont(":/font/Noto_Sans_SC/NotoSansSC-Light.otf");
    QFontDatabase::addApplicationFont(":/font/Noto_Sans_SC/NotoSansSC-Medium.otf");
    QFontDatabase::addApplicationFont(":/font/Noto_Sans_SC/NotoSansSC-Regular.otf");
    QFontDatabase::addApplicationFont(":/font/Noto_Sans_SC/NotoSansSC-Thin.otf");

    QFontDatabase::addApplicationFont(":/font/PuHuiTi/Alibaba-PuHuiTi-Bold.ttf");
    QFontDatabase::addApplicationFont(":/font/PuHuiTi/Alibaba-PuHuiTi-Heavy.ttf");
    QFontDatabase::addApplicationFont(":/font/PuHuiTi/Alibaba-PuHuiTi-Light.ttf");
    QFontDatabase::addApplicationFont(":/font/PuHuiTi/Alibaba-PuHuiTi-Medium.ttf");
    QFontDatabase::addApplicationFont(":/font/PuHuiTi/Alibaba-PuHuiTi-Regular.ttf");

    QFontDatabase::addApplicationFont(":/font/Segoe/SEGOEUI.TTF");

    QSettings settings(zsCompanyName, zsEditor);
    QVariant use_chinese = settings.value("use_chinese");
    bool bCN = !use_chinese.isNull() && use_chinese.toBool();
    if (bCN) {
        QFont font("Alibaba PuHuiTi", 10);
        //QFont font("Noto Sans SC", 10);
        setFont(font);
    } else {
        QFont font("Segoe UI", 10);
        setFont(font);
    }
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
