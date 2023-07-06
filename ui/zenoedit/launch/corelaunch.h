#ifndef __CORE_LAUNCHER_H__
#define __CORE_LAUNCHER_H__

#include <QtWidgets>

class IGraphsModel;

struct LAUNCH_PARAM {
    int beginFrame = 0;
    int endFrame = 0;
    bool applyLightAndCameraOnly = false;
    bool applyMaterialOnly = false;

    bool enableCache = false;
    bool tempDir = false;
    QString cacheDir = "";
    int cacheNum = 1;
    bool autoRmCurcache = false;
};

void launchProgram(IGraphsModel *pModel, LAUNCH_PARAM param);
bool initZenCache(char* cachedir);
void killProgram();

#endif
