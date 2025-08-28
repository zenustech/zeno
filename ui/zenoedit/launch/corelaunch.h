#ifndef __CORE_LAUNCHER_H__
#define __CORE_LAUNCHER_H__

#include <QtWidgets>

class IGraphsModel;

enum runType {
    LoadAsset = 0,
    RunALL,
    RunMatrix,
    RunLightCamera,
    RunMaterial
}; 

struct LAUNCH_PARAM {
    int beginFrame = 0;
    int endFrame = 0;
    runType runtype = LoadAsset;
    bool always = false;
    QString generator;   //triggered by GenerateCommands Node to start a calc, to get the commands.
    bool enableCache = false;
    bool tempDir = false;
    QString cacheDir = "";
    QString objCacheDir = "";
    int cacheNum = 1;
    bool autoRmCurcache = false;    //auto remove cache when recording
    bool autoCleanCacheInCacheRoot = true;    //点击运行/退出时自动清理cache
    bool cmdRmHistoryCacheBeforeRun = false;   //运行前清理历史cache
    QString zsgPath;
    int projectFps = 24;
    QString paramPath;
    QString paramBase64;

    bool fromCmd = false;
};

void launchProgram(IGraphsModel *pModel, LAUNCH_PARAM param);
bool initZenCache(char* cachedir, int& cacheNum);
void killProgram();

#endif
