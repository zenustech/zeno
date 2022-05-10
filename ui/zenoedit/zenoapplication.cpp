#include <zenoio/reader/zsgreader.h>
#include "model/graphsmodel.h"
#include "zenoapplication.h"
#include "graphsmanagment.h"
#include "zenomainwindow.h"
#include <zeno/utils/log.h>


ZenoApplication::ZenoApplication(int &argc, char **argv)
    : QApplication(argc, argv)
    , m_pGraphs(new GraphsManagment(this))
    , m_bIOProcessing(false)
    , m_errSteam(std::clog)
{
    initFonts();
    initStyleSheets();
    m_errSteam.registerMsgHandler();
    zeno::log_info("build date: {} {}", __DATE__, __TIME__);
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
    Q_ASSERT(ret);
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

QSharedPointer<GraphsManagment> ZenoApplication::graphsManagment() const
{
    return m_pGraphs;
}

QStandardItemModel* ZenoApplication::logModel() const
{
    return m_pGraphs->logModel();
}

void ZenoApplication::setIOProcessing(bool bIOProcessing)
{
    m_bIOProcessing = bIOProcessing;
}

bool ZenoApplication::IsIOProcessing() const
{
    return m_bIOProcessing;
}

ZenoMainWindow* ZenoApplication::getMainWindow()
{
	foreach(QWidget * widget, topLevelWidgets())
		if (ZenoMainWindow* mainWindow = qobject_cast<ZenoMainWindow*>(widget))
			return mainWindow;
	return nullptr;
}
