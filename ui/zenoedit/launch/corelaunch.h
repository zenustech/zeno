#ifndef __CORE_LAUNCHER_H__
#define __CORE_LAUNCHER_H__

#include <QtWidgets>

const int TCP_PORT = 8887;

class GraphsModel;

void launchProgram(GraphsModel* pModel, int beginFrame, int endFrame);
void killProgram();

#endif
