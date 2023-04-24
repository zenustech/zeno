#include "calculateworker.h"
#include <zeno/extra/GlobalState.h>
#include <zeno/utils/scope_exit.h>
#include <zeno/extra/GlobalComm.h>
#include <zeno/extra/GlobalStatus.h>
#include <zeno/utils/logger.h>
#include <zeno/core/Graph.h>
#include <zeno/zeno.h>
#include <zeno/types/StringObject.h>
#include "settings/zsettings.h"
#include "cache/zcachemgr.h"
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include "zassert.h"


CalculateWorker::CalculateWorker(QObject *parent)
    : QObject(parent)
    , m_state(kStopped)
{
}

void CalculateWorker::setProgJson(const QString& json)
{
    progJson = json;
}

CalculateWorker::ProgramState CalculateWorker::state() const
{
    return m_state;
}

bool CalculateWorker::chkfail()
{
    auto globalStatus = zeno::getSession().globalStatus.get();
    if (globalStatus->failed()) {
        reportStatus(*globalStatus);
        return true;
    }
    return false;
}

void CalculateWorker::stop()
{
    m_state = kQuiting;
}

void CalculateWorker::reportStatus(zeno::GlobalStatus const& stat)
{
    if (!stat.failed())
        return;
    zeno::log_error("error in {}, message {}", stat.nodeName, stat.error->message);
    auto nodeName = stat.nodeName.substr(0, stat.nodeName.find(':'));
    emit errorReported(QString::fromStdString(nodeName), QString::fromStdString(stat.error->message));
}

bool CalculateWorker::initZenCache()
{
    QSettings settings(zsCompanyName, zsEditor);
    bool bEnableCache = settings.value("zencache-enable").toBool();
    bool bAutoRemove = settings.value("zencache-autoremove", true).toBool();
    QString finalPath;
    if (bEnableCache)
    {
        const QString& cacheRootdir = settings.value("zencachedir").toString();
        QDir dirCacheRoot(cacheRootdir);
        if (!QFileInfo(cacheRootdir).isDir() && !bAutoRemove)
        {
            zeno::log_warn("The path of cache is invalid, please choose another path.");
            return false;
        }

        std::shared_ptr<ZCacheMgr> mgr = zenoApp->getMainWindow()->cacheMgr();
        ZASSERT_EXIT(mgr, false);
        bool ret = mgr->initCacheDir(bAutoRemove, cacheRootdir);
        ZASSERT_EXIT(ret, false);
        finalPath = mgr->cachePath();
        int cnum = settings.value("zencachenum").toInt();

        zeno::getSession().globalComm->frameCache(finalPath.toStdString(), cnum);
    }
    else
    {
        zeno::getSession().globalComm->frameCache("", 0);
    }
    return bEnableCache;
}

void CalculateWorker::work()
{
    zeno::log_debug("launching program...");
    zeno::log_debug("program JSON: {}", progJson.toUtf8());

    auto session = &zeno::getSession();
    session->globalComm->clearState();
    session->globalState->clearState();
    session->globalStatus->clearState();

    bool bZenCache = initZenCache();

    auto graph = session->createGraph();
    graph->loadGraph(progJson.toUtf8());

    if (chkfail())
        return;
    if (m_state == kQuiting)
        return;

    session->globalComm->frameRange(graph->beginFrameNumber, graph->endFrameNumber);
    for (int frame = graph->beginFrameNumber; frame <= graph->endFrameNumber; frame++)
    {
        zeno::log_debug("begin frame {}", frame);
        session->globalState->frameid = frame;
        session->globalComm->newFrame();
        //corresponding to processPacket in viewdecode.cpp

        emit viewUpdated("newFrame");

        session->globalState->frameBegin();
        while (session->globalState->substepBegin()) {
            if (m_state == kQuiting)
                return;
            graph->applyNodesToExec();
            session->globalState->substepEnd();
            if (chkfail())
                return;
        }
        if (m_state == kQuiting)
            return;
        session->globalState->frameEnd();
        if (bZenCache)
            session->globalComm->dumpFrameCache(frame);
        session->globalComm->finishFrame();

        //test update.
        //Sleep(4000);

        emit viewUpdated("finishFrame");

        zeno::log_debug("end frame {}", frame);
        if (chkfail())
            return;
    }
    if (session->globalStatus->failed()) {
        reportStatus(*session->globalStatus);
    }
    zeno::log_debug("program finished");
    emit finished();
}