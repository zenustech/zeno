#include "zcachemgr.h"
#include "zassert.h"
#include <zeno/extra/GlobalComm.h>
#include <zeno/zeno.h>

ZCacheMgr::ZCacheMgr()
    : m_bTempDir(true)
    , m_cacheSeparate(false)
    , m_dirCreated(false)
{
}

bool ZCacheMgr::initCacheDir(bool bTempDir, QDir dirCacheRoot)
{
    if (m_cacheSeparate && m_dirCreated) {
        return true;
    }
    m_bTempDir = bTempDir;
    if (m_bTempDir)
    {
        m_spTmpCacheDir.reset(new QTemporaryDir);
        m_spTmpCacheDir->setAutoRemove(true);
        m_dirCreated = true;
    }
    else
    {
        QString tempDirPath = QDateTime::currentDateTime().toString("yyyy-MM-dd hh-mm-ss");
        bool ret = dirCacheRoot.mkdir(tempDirPath);
        ZASSERT_EXIT(ret, false);
        m_spCacheDir = dirCacheRoot;
        ret = m_spCacheDir.cd(tempDirPath);
        ZASSERT_EXIT(ret, false);
        m_dirCreated = true;
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

void ZCacheMgr::cacheSeparately(bool separate) {
    m_cacheSeparate = separate;
}

void ZCacheMgr::setDirCreated(bool dirCreated) {
    m_dirCreated = dirCreated;
}
