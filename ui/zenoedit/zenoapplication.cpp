#include <zenoio/reader/zsgreader.h>
#include "model/graphsmodel.h"
#include "zenoapplication.h"
#include "graphsmanagment.h"
#include "zenomainwindow.h"


ZenoApplication::ZenoApplication(int &argc, char **argv)
    : QApplication(argc, argv)
    , m_pGraphs(new GraphsManagment(this))
{
    initFonts();
    initStyleSheets();
}

ZenoApplication::~ZenoApplication()
{
}

void ZenoApplication::initStyleSheets()
{
    QByteArray bytes;
    QString qss;

	QFile file(":/stylesheet/qlabel.qss");
	bool ret = file.open(QIODevice::ReadOnly | QIODevice::Text);
    Q_ASSERT(ret);
    qss = file.readAll();

    file.setFileName(":/stylesheet/qlineedit.qss");
    ret = file.open(QIODevice::ReadOnly | QIODevice::Text);
    Q_ASSERT(ret);
    qss += file.readAll();

    file.setFileName(":/stylesheet/menu.qss");
    ret = file.open(QIODevice::ReadOnly | QIODevice::Text);
    Q_ASSERT(ret);
    qss += file.readAll();

    file.setFileName(":/stylesheet/qcombobox.qss");
    ret = file.open(QIODevice::ReadOnly | QIODevice::Text);
    Q_ASSERT(ret);
    qss += file.readAll();

    file.setFileName(":/stylesheet/qwidget.qss");
    ret = file.open(QIODevice::ReadOnly | QIODevice::Text);
    Q_ASSERT(ret);
    qss += file.readAll();

    file.setFileName(":/stylesheet/pushbutton.qss");
    ret = file.open(QIODevice::ReadOnly | QIODevice::Text);
    Q_ASSERT(ret);
    qss += file.readAll();

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

    QFontDatabase::addApplicationFont(":/font/HarmonyOS_Sans_SC/HarmonyOS_Sans_SC_Black.ttf");
    QFontDatabase::addApplicationFont(":/font/HarmonyOS_Sans_SC/HarmonyOS_Sans_SC_Bold.ttf");
    QFontDatabase::addApplicationFont(":/font/HarmonyOS_Sans_SC/HarmonyOS_Sans_SC_Light.ttf");
    QFontDatabase::addApplicationFont(":/font/HarmonyOS_Sans_SC/HarmonyOS_Sans_SC_Medium.ttf");
    QFontDatabase::addApplicationFont(":/font/HarmonyOS_Sans_SC/HarmonyOS_Sans_SC_Regular.ttf");
    QFontDatabase::addApplicationFont(":/font/HarmonyOS_Sans_SC/HarmonyOS_Sans_SC_Thin.ttf");
}

QSharedPointer<GraphsManagment> ZenoApplication::graphsManagment() const
{
    return m_pGraphs;
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