#include "zcachemgr.h"
#include "zassert.h"

ZCacheMgr::ZCacheMgr()
    : m_bTempDir(true)
    , m_isNew(true)
    , m_cacheOpt(Opt_Undefined)
{
}

bool ZCacheMgr::initCacheDir(bool bTempDir, QDir dirCacheRoot)
{
    if (!m_isNew && (m_cacheOpt == Opt_RunLightCameraMaterial || m_cacheOpt == Opt_AlwaysOn)) {
        return true;
    }
    m_bTempDir = bTempDir;
    if (m_bTempDir) {
        m_spTmpCacheDir.reset(new QTemporaryDir);
        m_spTmpCacheDir->setAutoRemove(true);
        m_isNew = false;
    } else {
        QString tempDirPath = QDateTime::currentDateTime().toString("yyyy-MM-dd hh-mm-ss");
        bool ret = dirCacheRoot.mkdir(tempDirPath);
        if (ret) {
            m_spCacheDir = dirCacheRoot;
            ret = m_spCacheDir.cd(tempDirPath);
            ZASSERT_EXIT(ret, false);
            m_isNew = false;
        }
    }
    return true;
}

QString ZCacheMgr::cachePath() const
{
    if (m_bTempDir)
    {
        return m_spTmpCacheDir->path();
    }
    else
    {
        return m_spCacheDir.path();
    }
}

std::shared_ptr<QTemporaryDir> ZCacheMgr::getTempDir() const
{
    return m_spTmpCacheDir;
}

QDir ZCacheMgr::getPersistenceDir() const
{
    return m_spCacheDir;
}


void ZCacheMgr::setCacheOpt(cacheOption opt) {
    m_cacheOpt = opt;
}

void ZCacheMgr::setNewCacheDir(bool setNew) {
    m_isNew = setNew;
}

ZCacheMgr::cacheOption ZCacheMgr::getCacheOption() {
    return m_cacheOpt;
}
