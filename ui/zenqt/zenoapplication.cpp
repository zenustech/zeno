#include <zeno/io/zsg2reader.h>
#include "zenoapplication.h"
#include "model/graphsmanager.h"
#include "zenomainwindow.h"
#include <zeno/utils/log.h>
#include "util/log.h"
#include "startup/zstartup.h"
#include <style/zenostyle.h>
#include "settings/zenosettingsmanager.h"
#include "calculation/calculationmgr.h"
#include "uicommon.h"


ZenoApplication::ZenoApplication(int &argc, char **argv)
    : QApplication(argc, argv)
    , m_bUIApp(true)
    , m_calcMgr(new CalculationMgr(this))
{
    initMetaTypes();
    initFonts();
    initStyleSheets();
    verifyVersion();

    QStringList locations;
    locations = QStandardPaths::standardLocations(QStandardPaths::AppDataLocation);
#ifdef Q_OS_WIN
    locations = locations.filter("ProgramData");
    ZASSERT_EXIT(!locations.isEmpty());
    m_appDataPath.setPath(locations[0]);
#endif

    m_bUIApp = argc == 1;

    //only main editor needs this.
    if (m_bUIApp) {
        m_spUILogStream = std::make_shared<ZWidgetErrStream>(std::clog);
        m_spUILogStream->registerMsgHandler();
        //register thread log proxy
        connect(m_spUILogStream->threadLogProxy().get(), SIGNAL(threadlogReady(const QString&)), 
            this, SLOT(onThreadLogReady(const QString&)), Qt::QueuedConnection);
    }

    m_spProcClipboard = std::make_shared<ProcessClipboard>();
}

ZenoApplication::~ZenoApplication()
{
}

void ZenoApplication::onThreadLogReady(const QString& msg)
{
    if (msg.startsWith("["))
    {
        QMessageLogger logger("zeno", 0, 0);
        QChar tip = msg.at(1);

        auto& mgr = GraphsManager::instance();
        if (tip == 'T') {
            mgr.appendLog(QtDebugMsg, "zeno", 0, msg);
        }
        else if (tip == 'D') {
            mgr.appendLog(QtDebugMsg, "zeno", 0, msg);
        }
        else if (tip == 'I') {
            mgr.appendLog(QtInfoMsg, "zeno", 0, msg);
        }
        else if (tip == 'C') {
            mgr.appendLog(QtCriticalMsg, "zeno", 0, msg);
        }
        else if (tip == 'W') {
            mgr.appendLog(QtWarningMsg, "zeno", 0, msg);
        }
        else if (tip == 'E') {
            mgr.appendLog(QtFatalMsg, "zeno", 0, msg);
        }
    }
    else {
        auto& mgr = GraphsManager::instance();
        mgr.appendLog(QtDebugMsg, "zeno", 0, msg);
    }
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

void ZenoApplication::initMetaTypes() 
{
    QMetaType::registerEqualsComparator<UI_VECTYPE>();
    QMetaType::registerEqualsComparator<CURVES_DATA>();
    qRegisterMetaType<NodeState>();
    qRegisterMetaType<zeno::ObjPath>();
    qRegisterMetaType<zeno::CustomUI>();
    qRegisterMetaType<zeno::reflect::Any>();
}

void ZenoApplication::initStyleSheets()
{
#if 0
    QFile f(":qdarkstyle/dark/darkstyle.qss");
    if (!f.exists()) {
        printf("Unable to set stylesheet, file not found\n");
    }
    else {
        f.open(QFile::ReadOnly | QFile::Text);
        QTextStream ts(&f);
        setStyleSheet(ts.readAll());
    }
#else
    QString qss;
    qss += readQss(":/stylesheet/qlabel.qss");
    qss += readQss(":/stylesheet/qlineedit.qss");
    qss += readQss(":/stylesheet/menu.qss");
    qss += readQss(":/stylesheet/qcombobox.qss");
    qss += readQss(":/stylesheet/pushbutton.qss");
    qss += readQss(":/stylesheet/scrollbar.qss");
    qss += readQss(":/stylesheet/spinbox.qss");
    qss += readQss(":/stylesheet/mainwindow.qss");
    qss += readQss(":/stylesheet/checkbox.qss");
    qss += readQss(":/stylesheet/qwidget.qss");
    qss += readQss(":/stylesheet/tabwidget.qss");
    setStyleSheet(qss);
#endif
}

void ZenoApplication::initFonts()
{
    //QFontDatabase::addApplicationFont(":/font/Noto_Sans_SC/NotoSansSC-Black.otf");
    //QFontDatabase::addApplicationFont(":/font/Noto_Sans_SC/NotoSansSC-Bold.otf");
    //QFontDatabase::addApplicationFont(":/font/Noto_Sans_SC/NotoSansSC-Light.otf");
    //QFontDatabase::addApplicationFont(":/font/Noto_Sans_SC/NotoSansSC-Medium.otf");
    //QFontDatabase::addApplicationFont(":/font/Noto_Sans_SC/NotoSansSC-Regular.otf");
    //QFontDatabase::addApplicationFont(":/font/Noto_Sans_SC/NotoSansSC-Thin.otf");

    //QFontDatabase::addApplicationFont(":/font/PuHuiTi/Alibaba-PuHuiTi-Bold.ttf");
    //QFontDatabase::addApplicationFont(":/font/PuHuiTi/Alibaba-PuHuiTi-Heavy.ttf");
    //QFontDatabase::addApplicationFont(":/font/PuHuiTi/Alibaba-PuHuiTi-Light.ttf");
    //QFontDatabase::addApplicationFont(":/font/PuHuiTi/Alibaba-PuHuiTi-Medium.ttf");
    //QFontDatabase::addApplicationFont(":/font/PuHuiTi/Alibaba-PuHuiTi-Regular.ttf");

    //QFontDatabase::addApplicationFont(":/font/Segoe/SEGOEUI.TTF");

    //QSettings settings(zsCompanyName, zsEditor);
    //QVariant use_chinese = settings.value("use_chinese");
    //bool bCN = !use_chinese.isNull() && use_chinese.toBool();
    //if (bCN) {
    //    QFont font("Alibaba PuHuiTi", 10);
    //    //QFont font("Noto Sans SC", 10);
    //    setFont(font);
    //} else {
    //    QFont font("Segoe UI", 10);
    //    setFont(font);
    //}
    //QFontDatabase::addApplicationFont(":/font/MiSans/MiSans-Thin.ttf");
    //QFontDatabase::addApplicationFont(":/font/MiSans/MiSans-Semibold.ttf");
    //QFontDatabase::addApplicationFont(":/font/MiSans/MiSans-Regular.ttf");
    //QFontDatabase::addApplicationFont(":/font/MiSans/MiSans-Normal.ttf");
    //QFontDatabase::addApplicationFont(":/font/MiSans/MiSans-Medium.ttf");
    //QFontDatabase::addApplicationFont(":/font/MiSans/MiSans-Light.ttf");
    //QFontDatabase::addApplicationFont(":/font/MiSans/MiSans-Heavy.ttf");
    //QFontDatabase::addApplicationFont(":/font/MiSans/MiSans-ExtraLight.ttf");
    //QFontDatabase::addApplicationFont(":/font/MiSans/MiSans-Demibold.ttf");
    //QFontDatabase::addApplicationFont(":/font/MiSans/MiSans-Bold.ttf");
    QFont font("Microsoft Sans Serif", 10);
    setFont(font);
}

GraphsManager* ZenoApplication::graphsManager() const
{
    return &GraphsManager::instance();
}

CalculationMgr* ZenoApplication::calculationMgr() const
{
    return m_calcMgr;
}

std::shared_ptr<ProcessClipboard> ZenoApplication::procClipboard() const
{
    return m_spProcClipboard;
}

QStandardItemModel* ZenoApplication::logModel() const
{
    return graphsManager()->logModel();
}

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
