#include "zcachemgr.h"
#include "zassert.h"
#include <zeno/funcs/ParseObjectFromUi.h>
#include <zeno/extra/GlobalComm.h>
#include <zenomodel/include/graphsmanagment.h>
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include <zeno/extra/GlobalStatus.h>


ZCacheMgr::ZCacheMgr()
    : m_bTempDir(true)
    , m_isNew(true)
{
    m_objTmpCacheDir.setAutoRemove(true);
}

bool ZCacheMgr::initCacheDir(QDir dirCacheRoot, LAUNCH_PARAM& param)
{
    initToViewNodesId();

    if (nextRunSkipCreateDir(param)) {
        return true;
    }

    if (param.tempDir || param.autoCleanCacheInCacheRoot)
        cleanCacheDir();

    m_bTempDir = param.tempDir;
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
                info.fileName().right(4) != ".vdb" &&
                info.fileName().left(sLen) != zeno::iotags::sZencache_lockfile_prefix)    //not zencache file or vdb file or cachelock file
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

bool ZCacheMgr::nextRunSkipCreateDir(LAUNCH_PARAM& param)
{
    if (param.reRun)
    {
        return false;
    }
    else if (m_bTempDir != param.tempDir)    //switch cache mode(tmp->notmp or notmp->tmp)
    {
        return false;
    }
    else if (!param.tempDir && lastRunCachePath.path() != "." && lastRunCachePath.path().mid(0, lastRunCachePath.path().lastIndexOf("/")) != param.cacheDir)    //cacheroot changed
    {
        return false;
    }
    else if(m_isNew || !lastRunCachePath.exists()) {    //open new file or last not exist
        return false;
    }
    else if(param.beginFrame != param.endFrame){
        QDir last = QDir(lastRunCachePath);
        last.setFilter(QDir::AllDirs | QDir::NoSymLinks | QDir::NoDotAndDotDot);
        for (int frame = param.beginFrame; frame <= param.endFrame; frame++)
            if (! QDir(last.path() + "/" + QString::number(1000000 + frame).right(6)).exists()) //incomplete cache
                return false;
        auto& globalComm = zeno::getSession().globalComm;
        if (globalComm->numOfFinishedFrame() < (param.endFrame - param.beginFrame + 1) && !zeno::getSession().globalStatus->error)
            return false;
    }
    return true;
}

bool ZCacheMgr::nodeCacheExist(QString& id, bool isStatic)
{
    ZenoMainWindow* mainWin = zenoApp->getMainWindow();
    ZASSERT_EXIT(mainWin, false);
    if (lastRunCachePath.path() != ".")
    {
        QDir framPath = lastRunCachePath;
        if (isStatic)
        {
            if (framPath.cd("_static"))
                if (QFile(framPath.path() + "/" + id + ".zencache").exists())
                    return true;
        }
        else {
            if (framPath.cd(QString::number(1000000 + mainWin->timelineInfo().currFrame).right(6)))
                if (QFile(framPath.path() + "/" + id + ".zencache").exists())
                    return true;
        }
    }
    return false;
}

void ZCacheMgr::initToViewNodesId()
{
    IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
    if (!pModel)
        return;
    std::map<std::string, bool> toViewNodes;
    QModelIndex subgIdx = pModel->index("main");
    for (int i = 0; i < pModel->itemCount(subgIdx); i++)
    {
        const QModelIndex& idx = pModel->index(i, subgIdx);
        if (idx.data(ROLE_OPTIONS).toInt() & OPT_VIEW)
            toViewNodes.insert(std::make_pair(idx.data(ROLE_OBJID).toString().toStdString(), idx.data(ROLE_OPTIONS).toInt() & OPT_ONCE));
    }
    zeno::getSession().globalComm->setToViewNodes(toViewNodes);
}
