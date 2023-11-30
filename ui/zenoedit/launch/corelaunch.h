#ifndef __CORE_LAUNCHER_H__
#define __CORE_LAUNCHER_H__

#include <QtWidgets>

class IGraphsModel;

struct LAUNCH_PARAM {
    int beginFrame = 0;
    int endFrame = 0;

    bool enableCache = false;
    bool tempDir = false;
    QString cacheDir = "";
    QString objCacheDir = "";
    int cacheNum = 1;
    bool autoRmCurcache = false;    //auto remove cache when recording
    bool autoCleanCacheInCacheRoot = true;    //auto remove cachedir in cache root
    QString zsgPath;
    int projectFps = 24;
    QString paramPath;
};

void launchProgram(IGraphsModel *pModel, LAUNCH_PARAM param);
bool initZenCache(char* cachedir, int& cacheNum);
void killProgram();

#endif
