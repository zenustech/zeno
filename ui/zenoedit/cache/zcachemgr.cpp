#include "zcachemgr.h"
#include "zassert.h"
#include <zeno/extra/GlobalComm.h>
#include <zeno/zeno.h>

ZCacheMgr::ZCacheMgr()
    : m_bTempDir(true)
    , m_separate(false)
    , m_normalObjDir("")
{
}

bool ZCacheMgr::initCacheDir(bool bTempDir, QDir dirCacheRoot)
{
    m_bTempDir = bTempDir;
    if (m_bTempDir)
    {
        m_spTmpCacheDir.reset(new QTemporaryDir);
        m_spTmpCacheDir->setAutoRemove(true);
    }
    else
    {
        QString tempDirPath = QDateTime::currentDateTime().toString("yyyy-MM-dd hh-mm-ss");
        bool ret = dirCacheRoot.mkdir(tempDirPath);
        ZASSERT_EXIT(ret, false);
        m_spCacheDir = dirCacheRoot;
        ret = m_spCacheDir.cd(tempDirPath);
        ZASSERT_EXIT(ret, false);
        if (!m_separate) {
            m_normalObjDir = m_spCacheDir.path();
        }
        else
        {
            zeno::getSession().globalComm->cacheNormalObjPath = m_normalObjDir.toStdString();
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

void ZCacheMgr::cacheSeparately(bool separate) {
    m_separate = separate;
}