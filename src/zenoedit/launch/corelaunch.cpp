#include "corelaunch.h"
#include <zenoui/model/graphsmodel.h>
#include <zenoui/model/modelrole.h>
#include <zenoio/writer/zsgwriter.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/utils/logger.h>
#include <zeno/core/Graph.h>
#include <zeno/zeno.h>
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
        STOPPED = 0,
        RUNNING,
        KILLING,
    };

    inline static std::mutex g_mtx;
    inline static std::atomic<ProgramState> g_state{STOPPED};

    std::string progJson;

#ifdef ZENO_MULTIPROCESS
    inline static std::unique_ptr<QProcess> g_proc;
#endif

    void operator()() const {
        std::unique_lock lck(g_mtx);
        start();
        zeno::log_debug("program finished");
        g_state = STOPPED;
    }

    void start() const {
        zeno::log_info("launching program...");
        zeno::log_debug("launching program JSON: {}", progJson);

#ifndef ZENO_MULTIPROCESS
        auto session = &zeno::getSession();
        session->globalState->clearState();

        auto graph = session->createGraph();
        graph->loadGraph(progJson.c_str());
        if (g_state == KILLING)
            return;

        auto nframes = graph->adhocNumFrames;
        for (int i = 0; i < nframes; i++) {
            session->globalState->frameBegin();
            while (session->globalState->substepBegin())
            {
                if (g_state == KILLING)
                    return;
                graph->applyNodesToExec();
                session->globalState->substepEnd();
            }
            if (g_state == KILLING)
                return;
            session->globalState->frameEnd();
        }
#else
        auto execDir = QCoreApplication::applicationDirPath().toStdString();
#if defined(Q_OS_WIN)
        auto runnerCmd = execDir + "\\zenorunner.exe";
#else
        auto runnerCmd = execDir + "/zenorunner";
#endif

        g_proc = std::make_unique<QProcess>();
        viewDecodeClear();
        g_proc->start(QString::fromStdString(runnerCmd), QStringList());
        while (!g_proc->waitForStarted()) {
            zeno::log_warn("still not started in 3s");
            if (g_state == KILLING)
                return;
        }

        std::vector<char> buf(1<<20);
        g_proc->setReadChannel(QProcess::ProcessChannel::StandardOutput);
        while (g_proc && !g_proc->atEnd()) {
            if (g_state == KILLING)
                return;
            qint64 redSize = g_proc->read(buf.data(), buf.size() / 2);
            zeno::log_warn("g_proc->read got {} bytes", redSize);
            if (redSize > 0) {
                viewDecodeAppend(buf.data(), redSize);
            }
        }
        if (g_state == KILLING)
            return;
        buf.clear();
        while (!g_proc->waitForFinished()) {
            zeno::log_warn("still not finished in 3s, terminating");
            g_proc->terminate();
            if (g_state == KILLING)
                return;
        }
        int code = g_proc->exitCode();
        g_proc = nullptr;
        zeno::log_info("runner process exited with {}", code);
#endif
    }
};

void launchProgramJSON(std::string progJson)
{
    std::unique_lock lck(ProgramRunData::g_mtx, std::try_to_lock);
    if (!lck.owns_lock()) {
        zeno::log_warn("A program is already running! Please kill first");
        return;
    }

    ProgramRunData::g_state = ProgramRunData::RUNNING;
    std::thread thr(ProgramRunData{std::move(progJson)});
    thr.detach();
}


void killProgramJSON()
{
    ProgramRunData::g_state = ProgramRunData::KILLING;
}

}

void launchProgram(GraphsModel* pModel, int nframes)
{
    QJsonArray ret;
    ret.push_back(QJsonArray({"setAdhocNumFrames", nframes}));
    serializeScene(pModel, ret);

    QJsonDocument doc(ret);
    std::string progJson = doc.toJson(QJsonDocument::Compact).toStdString();
    launchProgramJSON(std::move(progJson));
}

void killProgram() {
    killProgramJSON();
}
