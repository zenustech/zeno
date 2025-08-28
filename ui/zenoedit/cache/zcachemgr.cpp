#include "zcachemgr.h"
#include "zassert.h"
#include <zeno/funcs/ParseObjectFromUi.h>
#include <zeno/extra/GlobalComm.h>
#include "settings/zsettings.h"
#include <regex>


ZCacheMgr::ZCacheMgr()
    : m_bTempDir(true)
    , m_isNew(true)
    , m_cacheOpt(Opt_Undefined)
{
    m_objTmpCacheDir.setAutoRemove(true);
}

bool ZCacheMgr::initCacheDir(bool bTempDir, QDir dirCacheRoot, bool bAutoCleanCache)
{
    if (!m_isNew && (m_cacheOpt == Opt_RunLightCameraMaterial || m_cacheOpt == Opt_RunMatrix || m_cacheOpt == Opt_AlwaysOn)) {
         return true;
    }
    if ((bTempDir || bAutoCleanCache) &&
            m_cacheOpt != Opt_RunLightCameraMaterial && m_cacheOpt != Opt_RunMatrix &&
            m_cacheOpt != Opt_AlwaysOn)
    {
        cleanCacheDir();
    }

    m_bTempDir = bTempDir;
    if (m_bTempDir) {
        m_spTmpCacheDir.reset(new QTemporaryDir);
        m_spTmpCacheDir->setAutoRemove(true);
        lastRunCachePath = QDir(m_spTmpCacheDir->path());
        m_isNew = false;
    }
    else {
        //QString tempDirPath = QDateTime::currentDateTime().toString("yyyy-MM-dd hh-mm-ss");
        QString eternalFolder = "zeno__cache__file__storage__path_" + QString::number(QCoreApplication::applicationPid());
        QDir enternalFullPath = QDir::cleanPath(dirCacheRoot.absoluteFilePath(eternalFolder));
        if (enternalFullPath.exists()) {
            QFileInfo enternalFullPathinfo(dirCacheRoot.absoluteFilePath(eternalFolder));
            if (enternalFullPath == QDir::currentPath() ||
                enternalFullPath == QDir::current().absolutePath() ||
                enternalFullPath == QDir::rootPath() || !enternalFullPathinfo.isDir()) {
            }
            else {
                enternalFullPath.removeRecursively();
            }
        }
        bool ret = dirCacheRoot.mkdir(eternalFolder);
        if (ret) {
            m_spCacheDir = dirCacheRoot;
            ret = m_spCacheDir.cd(eternalFolder);
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


void ZCacheMgr::setCacheOpt(cacheOption opt) {
    m_cacheOpt = opt;
}

void ZCacheMgr::setNewCacheDir(bool setNew) {
    m_isNew = setNew;
    if (m_isNew)
    {
        removeObjTmpCacheDir();
    }
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
                info.fileName().left(sLen) != zeno::iotags::sZencache_lockfile_prefix &&
                info.fileName() != "stampInfo.txt" &&
                info.fileName() != "runInfo.txt")    //not zencache file or cachelock file or stampInfo.txt
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

void ZCacheMgr::procExitCleanUp()
{
	QSettings settings(zsCompanyName, zsEditor);
	bool autoClean = settings.value("zencache-autoclean").isValid() ? settings.value("zencache-autoclean").toBool() : true;
	bool autoRemove = settings.value("zencache-autoremove").isValid() ? settings.value("zencache-autoremove").toBool() : false;
    removeObjTmpCacheDir();
	if (autoClean || autoRemove)
	{
        cleanCacheDir();
	}
}

std::vector<std::filesystem::path> ZCacheMgr::historyCacheList(QDir dirCacheRoot, bool isTemp)
{
    if (isTemp) {
        return {};
    }
    static bool cleared = false;
    if (cleared) {
        return {};
    }
    cleared = true;

    std::vector<std::filesystem::path> lst;

    QFileInfo dirInfo(dirCacheRoot.absolutePath());
	if (dirCacheRoot.exists() && dirInfo.isDir() && dirInfo.isReadable()) {
        auto lastAbsoluteCacheDir = lastRunCachePath.absolutePath();
		std::regex pattern("^zeno__cache__file__storage__path_\\d+$");
        dirCacheRoot.setFilter(QDir::AllDirs | QDir::Hidden | QDir::NoSymLinks | QDir::NoDotAndDotDot);
		for (const QFileInfo& fileInfo : dirCacheRoot.entryInfoList()) {
			if (fileInfo.isDir()) {
				QString dirName = fileInfo.fileName();
				QString absoluteDir = fileInfo.absoluteFilePath();
				if (std::regex_match(dirName.toStdString(), pattern) && lastAbsoluteCacheDir != absoluteDir) {
                    lst.push_back(std::filesystem::absolute(std::filesystem::path(absoluteDir.toStdString())));
				}
			}
		}
	}
    return lst;
}
