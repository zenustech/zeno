#include <io/zsgreader.h>
#include <model/graphsmodel.h>
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
}

GraphsManagment* ZenoApplication::graphsManagment() const
{
    return m_pGraphs;
}