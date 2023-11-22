#ifndef __ZCACHEMGR_H__
#define __ZCACHEMGR_H__

#include <QtWidgets>

class ZCacheMgr
{
public:
    ZCacheMgr();
    bool initCacheDir(bool bTempDir, QDir dir, bool bAutoCleanCache);
    QString cachePath() const;
    QString objCachePath() const;
    std::shared_ptr<QTemporaryDir> getTempDir() const;
    QDir getPersistenceDir() const;

    void setNewCacheDir(bool setNew);
    void cleanCacheDir();
    bool hasCacheOnly(QDir dir, bool& empty);
    void removeObjTmpCacheDir();

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