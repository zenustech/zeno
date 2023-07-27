#ifndef __ZENO_COMMON_H__
#define __ZENO_COMMON_H__

#include <QModelIndex>

struct TIMELINE_INFO
{
    int beginFrame;
    int endFrame;
    int currFrame;
    bool bAlways;

    TIMELINE_INFO() : beginFrame(0), endFrame(0), currFrame(0), bAlways(false) {}
};

struct APP_SETTINGS
{
    TIMELINE_INFO timeline;
    //todo: other settings.
};

enum SearchType
{
    SEARCH_SUBNET = 1 << 0,
    SEARCH_NODECLS = 1 << 1,
    SEARCH_NODEID = 1 << 2,		// search node ident.
    SEARCH_ANNO = 1 << 3,
    SEARCH_ARGS = 1 << 4,       // all args.
    SEARCH_CUSTOM_NAME = 1 << 5, 
    SEARCHALL = SEARCH_NODECLS | SEARCH_NODEID | SEARCH_SUBNET | SEARCH_ANNO | SEARCH_ARGS | SEARCH_CUSTOM_NAME
};

enum SearchOpt
{
    SEARCH_FUZZ = 1 << 0,
    SEARCH_MATCH_EXACTLY = 1 << 1,
    SEARCH_CASE_SENSITIVE= 1 << 2,
};

struct SEARCH_RESULT
{
    SearchType type;
    QModelIndex targetIdx;  //node or subgraph index.
    QModelIndex subgIdx;
    QString socket;     //the socket/param which contains the result.
};

struct LiveObjectData{
    std::string verSrc = "";
    std::string camSrc = "";
    int verLoadCount = 0;
    int camLoadCount = 0;
};

struct ZENO_RECORD_RUN_INITPARAM {
    QString sZsgPath = "";
    bool bRecord = false;
    bool bOptix = false;    //is optix view.
    bool isExportVideo = false;
    bool needDenoise = false;
    int iFrame = 0;
    int iSFrame = 0;
    int iSample = 0;
    int iBitrate = 0;
    int iFps = 0;
    QString sPixel = "";
    QString sPath = "";
    QString audioPath = "";
    QString configFilePath = "";
    QString videoName = "";
    QString subZsg = "";
    bool exitWhenRecordFinish = false;
};

#endif