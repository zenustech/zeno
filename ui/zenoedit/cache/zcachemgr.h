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

    enum cacheOption {
        Opt_Undefined = 0,
        Opt_RunAll,
        Opt_RunLightCameraMaterial,
        Opt_AlwaysOn
    };
    void setCacheOpt(cacheOption opt);
    void setNewCacheDir(bool setNew);
    cacheOption getCacheOption();
    void cleanCacheDir();
    bool hasCacheOnly(QDir dir, bool& empty);

private:
    QTemporaryDir m_objTmpCacheDir;
    std::shared_ptr<QTemporaryDir> m_spTmpCacheDir;
    QDir m_spCacheDir;
    bool m_bTempDir;

    bool m_isNew;
    cacheOption m_cacheOpt;

    QDir lastRunCachePath;
};

#endif