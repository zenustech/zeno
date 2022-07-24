#ifndef __CORE_LAUNCHER_H__
#define __CORE_LAUNCHER_H__

#include <QtWidgets>

class IGraphsModel;

void launchProgram(IGraphsModel* pModel, int beginFrame, int endFrame);
void killProgram();

#endif
