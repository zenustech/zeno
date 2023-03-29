#ifndef __CORE_LAUNCHER_H__
#define __CORE_LAUNCHER_H__

#include <QtWidgets>

class IGraphsModel;

void launchProgram(IGraphsModel *pModel, int beginFrame, int endFrame, bool applyLightAndCameraOnly = false);
bool initZenCache(char* cachedir);
void killProgram();

#endif
