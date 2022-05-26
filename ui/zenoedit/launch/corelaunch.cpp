#include "corelaunch.h"
#include "model/graphsmodel.h"
#include <zenoui/model/modelrole.h>
#include <zenoui/util/jsonhelper.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/utils/scope_exit.h>
#include <zeno/extra/GlobalComm.h>
#include <zeno/extra/GlobalStatus.h>
#include <zeno/utils/logger.h>
#include <zeno/core/Graph.h>
#include <zeno/zeno.h>
#include "zenoapplication.h"
#include "ztcpserver.h"
#include "graphsmanagment.h"
#include "serialize.h"
#include <thread>
#include <mutex>
#include <atomic>
#ifdef ZENO_MULTIPROCESS
#include <QProcess>
#include "viewdecode.h"
#endif

namespace {

struct ProgramRunData {
    enum ProgramState {
        kStopped = 0,
        kRunning,
        kQuiting,
    };

    inline static std::mutex g_mtx;
    inline static std::atomic<ProgramState> g_state{kStopped};

    std::string progJson;

#ifdef ZENO_MULTIPROCESS
    inline static std::unique_ptr<QProcess> g_proc;
#endif

    void operator()() const {
        std::unique_lock lck(g_mtx);
        start();
#ifdef ZENO_MULTIPROCESS
        if (g_proc) {
            zeno::log_warn("terminating runner process");
            g_proc->terminate();
            g_proc->waitForFinished(-1);
            int code = g_proc->exitCode();
            g_proc = nullptr;
            zeno::log_info("runner process terminated with {}", code);
        }
#endif
        zeno::log_debug("program finished");
        g_state = kStopped;
    }

    void reportStatus(zeno::GlobalStatus const &stat) const {
        if (!stat.failed()) return;
        zeno::log_error("reportStatus: error in {}, message {}", stat.nodeName, stat.error->message);
        auto nodeName = stat.nodeName.substr(0, stat.nodeName.find(':'));
        zenoApp->graphsManagment()->appendErr(QString::fromStdString(nodeName),
                                              QString::fromStdString(stat.error->message));
    }

    bool chkfail() const {
        auto globalStatus = zeno::getSession().globalStatus.get();
        if (globalStatus->failed()) {
            reportStatus(*globalStatus);
            return true;
        }
        return false;
    }

    void start() const {
        zeno::log_info("launching program...");
        zeno::log_debug("program JSON: {}", progJson);

#ifndef ZENO_MULTIPROCESS
        auto session = &zeno::getSession();
        session->globalComm->clearState();
        session->globalState->clearState();
        session->globalStatus->clearState();

        auto graph = session->createGraph();
        graph->loadGraph(progJson.c_str());

        if (chkfail()) return;
        if (g_state == kQuiting) return;

        session->globalComm->frameRange(graph->beginFrameNumber, graph->endFrameNumber);
        for (int frame = graph->beginFrameNumber; frame <= graph->endFrameNumber; frame++) {
            zeno::log_debug("begin frame {}", frame);
            session->globalComm->newFrame();
            session->globalState->frameBegin();
            while (session->globalState->substepBegin())
            {
                if (g_state == kQuiting)
                    return;
                graph->applyNodesToExec();
                session->globalState->substepEnd();
                if (chkfail()) return;
            }
            if (g_state == kQuiting) return;
            session->globalState->frameEnd();
            session->globalComm->finishFrame();
            zeno::log_debug("end frame {}", frame);
            if (chkfail()) return;
        }
        if (session->globalStatus->failed()) {
            reportStatus(*session->globalStatus);
        }
#else
        //auto execDir = QCoreApplication::applicationDirPath().toStdString();
//#if defined(Q_OS_WIN)
        //auto runnerCmd = execDir + "\\zenorunner.exe";
//#else
        //auto runnerCmd = execDir + "/zenorunner";
//#endif

        g_proc = std::make_unique<QProcess>();
        g_proc->setInputChannelMode(QProcess::InputChannelMode::ManagedInputChannel);
        g_proc->setReadChannel(QProcess::ProcessChannel::StandardOutput);
        g_proc->setProcessChannelMode(QProcess::ProcessChannelMode::ForwardedErrorChannel);
        int sessionid = zeno::getSession().globalState->sessionid;
        g_proc->start(QCoreApplication::applicationFilePath(), QStringList({"-runner", QString::number(sessionid)}));
        if (!g_proc->waitForStarted(-1)) {
            zeno::log_warn("process failed to get started, giving up");
            return;
        }

        g_proc->write(progJson.data(), progJson.size());
        g_proc->closeWriteChannel();

        std::vector<char> buf(1<<20); // 1MB
        viewDecodeClear();
        zeno::scope_exit decodeFin{[] {
            viewDecodeFinish();
        }};

        while (g_proc->waitForReadyRead(-1)) {
            while (!g_proc->atEnd()) {
                if (g_state == kQuiting) return;
                qint64 redSize = g_proc->read(buf.data(), buf.size());
                zeno::log_debug("g_proc->read got {} bytes (ping test has 19)", redSize);
                if (redSize > 0) {
                    viewDecodeAppend(buf.data(), redSize);
                }
            }
            if (chkfail()) break;
            if (g_state == kQuiting) return;
        }
        zeno::log_debug("still not ready-read, assume exited");
        decodeFin.reset();

        buf.clear();
        g_proc->terminate();
        int code = g_proc->exitCode();
        g_proc = nullptr;
        zeno::log_info("runner process exited with {}", code);
#endif
    }
};

void launchProgramJSON(std::string progJson)
{
#ifdef ZENO_MULTIPROCESS
    ZTcpServer *pServer = zenoApp->getServer();
    if (pServer)
    {
        pServer->startProc(std::move(progJson));
    }
#else
    std::unique_lock lck(ProgramRunData::g_mtx, std::try_to_lock);
    if (!lck.owns_lock()) {
        zeno::log_warn("A program is already running! Please kill first");
        return;
    }

    ProgramRunData::g_state = ProgramRunData::kRunning;
    std::thread thr(ProgramRunData{std::move(progJson)});
    thr.detach();
#endif
}


void killProgramJSON()
{
    zeno::log_info("killing current program");
#ifdef ZENO_MULTIPROCESS
    ZTcpServer *pServer = zenoApp->getServer();
    if (pServer)
    {
        pServer->killProc();
    }
#else
    ProgramRunData::g_state = ProgramRunData::kQuiting;
#endif
}

}

void launchProgram(GraphsModel* pModel, int beginFrame, int endFrame)
{
	rapidjson::StringBuffer s;
	RAPIDJSON_WRITER writer(s);
    {
        JsonArrayBatch batch(writer);
        JsonHelper::AddVariantList({"setBeginFrameNumber", beginFrame}, "int", writer);
        JsonHelper::AddVariantList({"setEndFrameNumber", endFrame}, "int", writer);
        serializeScene(pModel, writer);
    }
    std::string progJson(s.GetString());
    launchProgramJSON(std::move(progJson));
}

void killProgram() {
    killProgramJSON();
}
