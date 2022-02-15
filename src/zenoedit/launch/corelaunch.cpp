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
#include "TinyProcessLib/process.hpp"
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


void waitProgramJSON()
{
    std::unique_lock lck(ProgramRunData::g_mtx);
}

void killProgramJSON()
{
    zeno::log_warn("cannot perform kill when ZENO_MULTIPROCESS is OFF");
}

#else

std::unique_ptr<TinyProcessLib::Process> g_proc;

void killProgramJSON();

void launchProgramJSON(std::string progJson)
{
    killProgramJSON();
    zeno::log_info("launching program JSON: {}", progJson);

    viewDecodeClear();
    auto execdir = QCoreApplication::applicationDirPath().toStdString();
#if defined(Q_OS_WIN)
    auto runnerCommand = "\"" + execdir + "\\zenorunner.exe" + "\"";
#else
    auto runnerCommand = "'" + execdir + "/zenorunner" + "'";
#endif
    g_proc = std::make_unique<TinyProcessLib::Process>
        ( /*command=*/runnerCommand
        , /*path=*/""
        , /*read_stdout=*/[] (const char *buf, size_t n) {
            viewDecodeAppend(buf, n);
        }
        , /*read_stderr=*/nullptr
        , /*open_stdin=*/true
        );
    g_proc->write(progJson.data(), progJson.size());
    g_proc->close_stdin();
}

void waitProgramJSON()
{
    if (!g_proc) return;
    int status = g_proc->get_exit_status();
    zeno::log_info("runner process exited with {}", status);
    g_proc = nullptr;
}

void killProgramJSON()
{
    if (g_proc) {
        zeno::log_info("killing existing runner process...");
        g_proc->kill(true);
        waitProgramJSON();
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
