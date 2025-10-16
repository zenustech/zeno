#include <zeno/io/zsg2reader.h>
#include "zenoapplication.h"
#include "model/graphsmanager.h"
#include "zenomainwindow.h"
#include <zeno/utils/log.h>
#include <zeno/core/Session.h>
#include "util/log.h"
#include "startup/zstartup.h"
#include <style/zenostyle.h>
#include "settings/zenosettingsmanager.h"
#include "calculation/calculationmgr.h"
#include "model/nodecatemodel.h"
#include "model/mouseutils.h"
#include "uicommon.h"
#include "declmetatype.h"
#include <QuickQanava>
#include <QQuickStyle>


static MouseUtils mouseUtils;

class WinEventFilter : public QAbstractNativeEventFilter
{
public:
    bool nativeEventFilter(const QByteArray& eventType, void* message, long* result) override {
        if (eventType == "windows_generic_MSG" || eventType == "windows_dispatcher_MSG") {
            MSG* msg = static_cast<MSG*>(message);
            if (msg->message >= zeno::SOVLER_START && msg->message <= zeno::SOLVER_END) {
                int lp = static_cast<int>(msg->lParam);
                int wp = static_cast<int>(msg->wParam);
                zenoApp->getMainWindow()->onSolverCallback((zeno::SOLVER_MSG)msg->message, std::min(lp, wp), std::max(lp, wp));
                return true;
            }
        }
        return false;
    }

};
static WinEventFilter win32Filter;




ZenoApplication::ZenoApplication(int &argc, char **argv)
    : QApplication(argc, argv)
    , m_bUIApp(true)
    , m_engine(nullptr)
    , m_calcMgr(new CalculationMgr(this))
    , m_graphsMgr(new GraphsManager(this))
    , m_nodecates(new NodeCateModel(this))
    , m_menuEventFilter(new MenuEventFilter(this))
{
    initMetaTypes();
    initFonts();
    initStyleSheets();
    verifyVersion();

    installNativeEventFilter(&win32Filter);

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
    zeno::getSession().destroy(); //在卸载插件前先清空所有图
}

void ZenoApplication::onThreadLogReady(const QString& msg)
{
    if (msg.startsWith("["))
    {
        QMessageLogger logger("zeno", 0, 0);
        QChar tip = msg.at(1);

        auto mgr = m_graphsMgr;
        if (tip == 'T') {
            mgr->appendLog(QtDebugMsg, "zeno", 0, msg);
        }
        else if (tip == 'D') {
            mgr->appendLog(QtDebugMsg, "zeno", 0, msg);
        }
        else if (tip == 'I') {
            mgr->appendLog(QtInfoMsg, "zeno", 0, msg);
        }
        else if (tip == 'C') {
            mgr->appendLog(QtCriticalMsg, "zeno", 0, msg);
        }
        else if (tip == 'W') {
            mgr->appendLog(QtWarningMsg, "zeno", 0, msg);
        }
        else if (tip == 'E') {
            mgr->appendLog(QtFatalMsg, "zeno", 0, msg);
        }
    }
    else {
        m_graphsMgr->appendLog(QtDebugMsg, "zeno", 0, msg);
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

void ZenoApplication::cleanQmlEngine() {
    delete m_engine;
    m_engine = nullptr;
}

void ZenoApplication::initMetaTypes() 
{
    QMetaType::registerEqualsComparator<UI_VECTYPE>();
    QMetaType::registerEqualsComparator<CURVES_DATA>();
    QMetaType::registerEqualsComparator<zeno::reflect::Any>();
    qRegisterMetaType<QmlNodeRunStatus::Value>();
    qRegisterMetaType<zeno::ObjPath>();
    qRegisterMetaType<zeno::CustomUI>();
    qRegisterMetaType<zeno::reflect::Any>();
    qRegisterMetaType<zeno::render_reload_info>();
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
    return m_graphsMgr;
}

NodeCateModel* ZenoApplication::nodecateModel() const {
    return m_nodecates;
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
    foreach(QWidget * widget, topLevelWidgets())
        if (ZenoMainWindow* mainWindow = qobject_cast<ZenoMainWindow*>(widget))
            return mainWindow;
    return nullptr;
}

void ZenoApplication::initQuickQanavas()
{
    if (!m_engine) {
        QQuickStyle::setStyle("Material");
        m_engine = new QQmlApplicationEngine(this);

        QSurfaceFormat format;
        format.setSamples(8);
        QSurfaceFormat::setDefaultFormat(format);

        m_engine->rootContext()->setContextProperty("nodecatesmodel", m_nodecates);
        m_engine->rootContext()->setContextProperty("MouseUtils", &mouseUtils);
        m_engine->rootContext()->setContextProperty("calcmgr", m_calcMgr);

        QuickQanava::initialize(m_engine);
        m_graphsMgr->initRootObjects();
    }
}

QQmlApplicationEngine* ZenoApplication::getQmlEngine() const {
    return m_engine;
}

QWidget *ZenoApplication::getWindow(const QString &objName)
{
    foreach (QWidget *widget, QApplication::allWidgets())
        if(widget->objectName() == objName)
            return widget;
    return nullptr;
}
