#include "zcachemgr.h"
#include "zassert.h"
#include <zeno/funcs/ParseObjectFromUi.h>
#include <zeno/extra/GlobalComm.h>
#include <zenomodel/include/graphsmanagment.h>
#include "zenoapplication.h"


ZCacheMgr::ZCacheMgr()
    : m_bTempDir(true)
    , m_isNew(true)
{
    m_objTmpCacheDir.setAutoRemove(true);
}

bool ZCacheMgr::initCacheDir(bool bTempDir, QDir dirCacheRoot, bool bAutoCleanCache)
{
    clearNotUsedToViewCache();

    if (!m_isNew)
        return true;

    if (bTempDir || bAutoCleanCache)
        cleanCacheDir();

    m_bTempDir = bTempDir;
    if (m_bTempDir) {
        m_spTmpCacheDir.reset(new QTemporaryDir);
        m_spTmpCacheDir->setAutoRemove(true);
        lastRunCachePath = QDir(m_spTmpCacheDir->path());
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

QString ZCacheMgr::objCachePath() const
{
    QString path = m_objTmpCacheDir.path();
    zeno::log_debug("node cache dir {}", path.toStdString());
    return path;
}

std::shared_ptr<QTemporaryDir> ZCacheMgr::getTempDir() const
{
    return m_spTmpCacheDir;
}

QDir ZCacheMgr::getPersistenceDir() const
{
    return m_spCacheDir;
}

void ZCacheMgr::setNewCacheDir(bool setNew) {
    m_isNew = setNew;
    if (m_isNew)
    {
        removeObjTmpCacheDir();
    }
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

void ZCacheMgr::removeObjTmpCacheDir()
{
    m_objTmpCacheDir.remove();
}

void ZCacheMgr::clearNotUsedToViewCache()
{
    IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
    if (!pModel)
        return;
    QModelIndex subgIdx = pModel->index("main");
    QSet<QString> newToViewNodes;
    for (int i = 0; i < pModel->itemCount(subgIdx); i++)
    {
        const QModelIndex& idx = pModel->index(i, subgIdx);
        if (idx.data(ROLE_OPTIONS).toInt() & OPT_VIEW)
            newToViewNodes.insert(idx.data(ROLE_OBJID).toString());
    }

    if (m_isNew)
    {
        lastRunToViewNodes.swap(newToViewNodes);
        return;
    }

    QVector<QString> toViewCachesToBeRemoved;
    for (auto id : lastRunToViewNodes)
        if (!newToViewNodes.contains(id))
            toViewCachesToBeRemoved.push_back(id);
    lastRunToViewNodes.swap(newToViewNodes);

    for (auto framePath : lastRunCachePath.entryInfoList()) {
        if (framePath.isDir()) {
            for (auto id : toViewCachesToBeRemoved) {
                QString toViewCacheToBeRemoved = framePath.absoluteFilePath() + "/" + id + ".zencache";
                if (QFile(toViewCacheToBeRemoved).exists())
                    lastRunCachePath.remove(toViewCacheToBeRemoved);
            }
        }
    }
}
