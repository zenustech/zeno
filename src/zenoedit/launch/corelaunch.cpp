#include "corelaunch.h"
#include <zenoui/model/graphsmodel.h>
#include <zenoui/model/modelrole.h>
#include <zenoio/writer/zsgwriter.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/utils/logger.h>
#include <zeno/zeno.h>
#include "serialize.h"
#include <thread>
#include <mutex>
#ifdef ZENO_MULTIPROCESS
#include <QProcess>
#include <QThread>
#include "viewdecode.h"

/*#ifdef _WIN32
#include <windows.h>
#undef min
#undef max
static std::string get_exec_path() {
	char szFilePath[MAX_PATH + 1] = { 0 };
	GetModuleFileNameA(NULL, szFilePath, MAX_PATH);
	(strrchr(szFilePath, '\\'))[0] = 0;
	string path = szFilePath;
	return path;
}
#define PATH_SEP '\\'
#define SH_QUOTE '"'
#else
#include <unistd.h>
#include <stdio.h>
static std::string get_exec_path() {
    auto link = "/proc/" + std::to_string(getpid()) + "/exe";
    ssize_t len = readlink(link.c_str(), NULL, 0);
    if (len < 0) {
        perror("readlink");
        return {};
    }
    std::string path;
    path.resize(len);
    if (readlink(link.c_str(), path.data(), len) < 0) {
        return {};
    }
	return path;
}
#define PATH_SEP '/'
#define SH_QUOTE '\''
#endif

static std::string get_exec_dir() {
    auto path = get_exec_path();
    return path.substr(0, path.find_last_of(PATH_SEP));
}*/
#endif

namespace {
#ifndef ZENO_MULTIPROCESS

struct ProgramRunData {
    inline static std::mutex g_mtx;

    std::string progJson;

    void operator()() const {
        std::unique_lock _(g_mtx);

        zeno::log_info("launching program JSON: {}", progJson);

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
    //std::unique_lock lck(ProgramRunData::g_mtx);
    zeno::log_warn("cannot perform kill when ZENO_MULTIPROCESS is OFF");
}

#else

std::unique_ptr<QProcess> g_proc;
std::mutex g_proc_mtx;

void killProgramJSON();

void launchProgramJSON(std::string progJson)
{
    killProgramJSON();
    zeno::log_info("launching program JSON: {}", progJson);

    viewDecodeClear();
    auto execDir = QCoreApplication::applicationDirPath().toStdString();
#if defined(Q_OS_WIN)
    auto runnerCmd = execDir + "\\zenorunner.exe";
#else
    auto runnerCmd = execDir + "/zenorunner";
#endif
    std::unique_lock lck(g_proc_mtx);
    if (g_proc) return;

    std::thread thr([runnerCmd = std::move(runnerCmd), progJson = std::move(progJson)] {
        g_proc = std::make_unique<QProcess>();
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
}

void killProgramJSON()//TODO: send to queue, QProcess cannot be called from another thread
{
    std::unique_lock lck(g_proc_mtx);
    if (g_proc) {
        zeno::log_info("killing existing runner process...");
        g_proc->terminate();
        g_proc->waitForFinished(-1);
        int code = g_proc->exitCode();
        zeno::log_info("killed runner process exited with {}", code);
        g_proc = nullptr;
    }
}

#endif
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
