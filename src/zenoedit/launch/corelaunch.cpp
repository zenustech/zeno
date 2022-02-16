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
#ifdef ZENO_MULTIPROCESS
#include <QProcess>
#include "viewdecode.h"
#endif

namespace {

struct ProgramRunData {
    inline static std::mutex g_mtx;

    std::string progJson;

#ifdef ZENO_MULTIPROCESS
    inline static std::unique_ptr<QProcess> g_proc;
#endif

    void operator()() const {
        std::unique_lock _(g_mtx);

        zeno::log_info("launching program JSON: {}", progJson);

#ifdef ZENO_MULTIPROCESS
        auto session = &zeno::getSession();
        session->globalState->clearState();

        auto graph = session->createGraph();
        graph->loadGraph(progJson.c_str());

        auto nframes = graph->adhocNumFrames;
        for (int i = 0; i < nframes; i++) {
            session->globalState->frameBegin();
            while (session->globalState->substepBegin())
            {
                graph->applyNodesToExec();
                session->globalState->substepEnd();
            }
            session->globalState->frameEnd();
        }
#else
        if (g_proc) return;

        std::thread thr([progJson = std::move(progJson)] {
            auto execDir = QCoreApplication::applicationDirPath().toStdString();
#if defined(Q_OS_WIN)
            auto runnerCmd = execDir + "\\zenorunner.exe";
#else
            auto runnerCmd = execDir + "/zenorunner";
#endif

            g_proc = std::make_unique<QProcess>();
            viewDecodeClear();
            g_proc->start(QString::fromStdString(runnerCmd), QStringList());
            if (!g_proc->waitForStarted()) {
                zeno::log_warn("still not started in 3s");
                g_proc->waitForStarted(-1);
            }

            std::vector<char> buf(1<<20);
            std::unique_lock lck(g_proc_mtx);
            g_proc->setReadChannel(QProcess::ProcessChannel::StandardOutput);
            while (g_proc && !g_proc->atEnd()) {
                qint64 redSize = g_proc->read(buf.data(), buf.size() / 2);
                lck.unlock();
                zeno::log_warn("g_proc->read got {} bytes", redSize);
                if (redSize > 0) {
                    viewDecodeAppend(buf.data(), redSize);
                }
                lck.lock();
            }
            buf.clear();
            if (!g_proc->waitForFinished()) {
                zeno::log_warn("still not finished in 3s, terminating");
                g_proc->terminate();
                g_proc->waitForFinished(-1);
            }
            int code = g_proc->exitCode();
            g_proc = nullptr;
            lck.unlock();
            zeno::log_info("runner process exited with {}", code);
        });
        thr.detach();
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

    std::thread thr(ProgramRunData{std::move(progJson)});
    thr.detach();
}


void killProgramJSON()
{
    //std::unique_lock lck(ProgramRunData::g_mtx);//TODO
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
