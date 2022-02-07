#ifndef __CORE_LAUNCHER_H__
#define __CORE_LAUNCHER_H__

#include <QtWidgets>

extern QString g_iopath;

class GraphsModel;

void launchProgram(GraphsModel* pModel, int nframes);

#endif
