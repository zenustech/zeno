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

private:
    std::shared_ptr<QTemporaryDir> m_spTmpCacheDir;
    QDir m_spCacheDir;
    bool m_bTempDir;
};

#endif