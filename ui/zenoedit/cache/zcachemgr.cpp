#include "zcachemgr.h"
#include "zassert.h"

ZCacheMgr::ZCacheMgr()
    : m_bAutoRemove(false)
    , m_subDir("")
    , m_isNew(true)
    , m_cacheOpt(Opt_Undefined)
{
}

bool ZCacheMgr::initCacheDir(bool bAutoRemove, QDir dirCacheRoot, QString subdir)
{
    if (!m_isNew && (m_cacheOpt == Opt_RunLightCameraMaterial || m_cacheOpt == Opt_AlwaysOnLightCameraMaterial)) {
        return true;
    }
    m_bAutoRemove = bAutoRemove;
    if (subdir.isEmpty()) {
        m_spTmpCacheDir.reset(new QTemporaryDir(dirCacheRoot.path() + "/"));
        m_spTmpCacheDir->setAutoRemove(m_bAutoRemove);
        m_isNew = false;
    } else {
        m_subDir = subdir;
        if (!QDir(dirCacheRoot.path() + "/" + m_subDir).exists())
        {
            bool ret = dirCacheRoot.mkdir(m_subDir);
            ZASSERT_EXIT(ret, false);
        }
        bool ret = dirCacheRoot.cd(m_subDir);
        ZASSERT_EXIT(ret, false);
        QString tempDirPath = QDateTime::currentDateTime().toString("yyyy-MM-dd hh-mm-ss");
        ret = dirCacheRoot.mkdir(tempDirPath);
        if (ret) {
            if (m_bAutoRemove)
            {
                m_spCacheDir.removeRecursively();
            }
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
    if (m_subDir.isEmpty())
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
