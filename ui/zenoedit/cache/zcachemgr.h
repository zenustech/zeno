#ifndef __ZCACHEMGR_H__
#define __ZCACHEMGR_H__

#include <QtWidgets>

class ZCacheMgr
{
public:
    ZCacheMgr();
    bool initCacheDir(bool bTempDir, QDir dir);
    QString cachePath() const;
    std::shared_ptr<QTemporaryDir> getTempDir() const;
    QDir getPersistenceDir() const;

    void cacheSeparately(bool separate);
    void setDirCreated(bool dirCreated);

private:
    std::shared_ptr<QTemporaryDir> m_spTmpCacheDir;
    QDir m_spCacheDir;
    bool m_bTempDir;

    bool m_cacheSeparate;
    bool m_dirCreated;
};

#endif