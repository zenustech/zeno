#include "zcachemgr.h"
#include "zassert.h"
#include <zeno/funcs/ParseObjectFromUi.h>
#include <zeno/extra/GlobalComm.h>


ZCacheMgr::ZCacheMgr()
    : m_bTempDir(true)
    , m_isNew(true)
    , m_cacheOpt(Opt_Undefined)
{
}

bool ZCacheMgr::initCacheDir(bool bTempDir, QDir dirCacheRoot, bool bAutoCleanCache)
{
    if (!m_isNew && (m_cacheOpt == Opt_RunLightCameraMaterial || m_cacheOpt == Opt_AlwaysOn)) {
         return true;
    }
    if (!bTempDir && bAutoCleanCache &&
        m_cacheOpt != Opt_RunLightCameraMaterial &&
        m_cacheOpt != Opt_AlwaysOn)
    {
        cleanCacheDir();
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
            lastRunCachePath = m_spCacheDir;
        }
    }
    return true;
}

QString ZCacheMgr::cachePath() const
{
    if (m_bTempDir)
    {
        if (m_spTmpCacheDir)
        {
            return m_spTmpCacheDir->path();
        }
        else {
            return "";
        }
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

void ZCacheMgr::cleanCacheDir()
{
    QString selfPath = QCoreApplication::applicationDirPath();

    bool dataTimeCacheDirEmpty = true;
    if (lastRunCachePath.exists() &&
        hasCacheOnly(lastRunCachePath, dataTimeCacheDirEmpty) &&
        !dataTimeCacheDirEmpty &&
        lastRunCachePath.path() != selfPath &&
        lastRunCachePath.path() != ".")
    {
        auto& pGlobalComm = zeno::getSession().globalComm;
        ZASSERT_EXIT(pGlobalComm);
        pGlobalComm->removeCachePath();
        zeno::log_info("remove dir: {}", lastRunCachePath.absolutePath().toStdString());
    }
    if (dataTimeCacheDirEmpty && QDateTime::fromString(lastRunCachePath.dirName(), "yyyy-MM-dd hh-mm-ss").isValid())
    {
        lastRunCachePath.rmdir(lastRunCachePath.path());
        zeno::log_info("remove dir: {}", lastRunCachePath.absolutePath().toStdString());
    }
}

bool ZCacheMgr::hasCacheOnly(QDir dir, bool& empty)
{
    bool bHasCacheOnly = true;
    dir.setFilter(QDir::AllDirs | QDir::Files | QDir::Hidden | QDir::NoSymLinks | QDir::NoDotAndDotDot);
    dir.setSorting(QDir::DirsLast);
    for (auto info : dir.entryInfoList())
    {
        if (info.isFile())
        {
            empty = false;
            size_t sLen = strlen(zeno::iotags::sZencache_lockfile_prefix);
            if (info.fileName().right(9) != ".zencache" &&
                info.fileName().left(sLen) != zeno::iotags::sZencache_lockfile_prefix)    //not zencache file or cachelock file
            {
                return false;
            }
        }
        else if (info.isDir())
        {
            if (!hasCacheOnly(info.filePath(), empty))
            {
                return false;
            }
        }
    }
    return bHasCacheOnly;
}
