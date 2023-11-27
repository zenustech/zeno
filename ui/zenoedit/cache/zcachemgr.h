#ifndef __ZCACHEMGR_H__
#define __ZCACHEMGR_H__

#include <QtWidgets>
#include "launch/corelaunch.h"

class ZCacheMgr
{
public:
    ZCacheMgr();
    bool initCacheDir(QDir dir, LAUNCH_PARAM& param);
    QString cachePath() const;
    QString objCachePath() const;
    std::shared_ptr<QTemporaryDir> getTempDir() const;
    QDir getPersistenceDir() const;

    void setNewCacheDir(bool setNew);
    void cleanCacheDir();
    bool hasCacheOnly(QDir dir, bool& empty);
    void removeObjTmpCacheDir();
    bool nextRunSkipCreateDir(LAUNCH_PARAM& param);

private:
    void clearNotUsedToViewCache();

    QTemporaryDir m_objTmpCacheDir;
    std::shared_ptr<QTemporaryDir> m_spTmpCacheDir;
    QDir m_spCacheDir;
    bool m_bTempDir;

    bool m_isNew;

    QDir lastRunCachePath;

    QSet<QString> lastRunToViewNodes;
};

#endif