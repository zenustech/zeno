#include <zenoio/reader/zsgreader.h>
#include "model/graphsmodel.h"
#include "zenoapplication.h"
#include "graphsmanagment.h"


ZenoApplication::ZenoApplication(int &argc, char **argv)
    : QApplication(argc, argv)
    , m_pGraphs(new GraphsManagment(this))
{
    initFonts();
}

ZenoApplication::~ZenoApplication()
{
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