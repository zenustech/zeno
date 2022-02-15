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
#endif

namespace {

struct ProgramRunData {
    inline static std::mutex g_mtx;

    std::string progJson;
    int nframes;

    void operator()() const {
        std::unique_lock _(g_mtx);

        zeno::log_info("launching program JSON: {}", progJson);

#ifdef ZENO_MULTIPROCESS
#else
        auto session = &zeno::getSession();
        session->globalState->clearState();

        auto graph = session->createGraph();
        graph->loadGraph(progJson.c_str());

        for (int i = 0; i < nframes; i++) {
            session->globalState->frameBegin();
            while (session->globalState->substepBegin())
            {
                graph->applyNodesToExec();
                session->globalState->substepEnd();
            }
            session->globalState->frameEnd();
        }
#endif
    }
};

}

void launchProgram(GraphsModel* pModel, int nframes)
{
    std::unique_lock lck(ProgramRunData::g_mtx, std::try_to_lock);
    if (!lck.owns_lock()) {
        zeno::log_warn("A program is already running! Please kill first");
        return;
    }

    QJsonArray ret;
    serializeScene(pModel, ret);

    QJsonDocument doc(ret);
    std::string progJson = doc.toJson(QJsonDocument::Compact).toStdString();

    std::thread thr(ProgramRunData{std::move(progJson), nframes});
    thr.detach();
}
