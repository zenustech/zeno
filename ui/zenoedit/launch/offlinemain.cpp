#include "corelaunch.h"
#include "serialize.h"
#include "util/log.h"
#include "graphsmanagment.h"
#include "model/graphsmodel.h"
#include <zenoui/util/jsonhelper.h>
#include <zenoui/model/modelrole.h>
#include <zeno/core/Session.h>
#include <zeno/utils/scope_exit.h>
#include <zeno/extra/GlobalComm.h>
#include <zeno/extra/GlobalStatus.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/utils/logger.h>
#include <zeno/core/Graph.h>

static int offline_start(const char *progJson) {
    zeno::log_trace("program JSON: {}", progJson);

    auto session = &zeno::getSession();
    session->globalComm->clearState();
    session->globalState->clearState();
    session->globalStatus->clearState();

    auto graph = session->createGraph();
    graph->loadGraph(progJson);

    auto chkfail = [&] {
        auto globalStatus = session->globalStatus.get();
        if (globalStatus->failed()) {
            zeno::log_error("error in {}, message {}", globalStatus->nodeName, globalStatus->error->message);
            return true;
        }
        return false;
    };

    if (chkfail()) return 1;

    session->globalComm->frameRange(graph->beginFrameNumber, graph->endFrameNumber);
    for (int frame = graph->beginFrameNumber; frame <= graph->endFrameNumber; frame++) {
        zeno::log_info("begin frame {}", frame);
        session->globalComm->newFrame();
        session->globalState->frameBegin();
        while (session->globalState->substepBegin())
        {
            graph->applyNodesToExec();
            session->globalState->substepEnd();
            if (chkfail()) return 1;
        }
        session->globalState->frameEnd();
        session->globalComm->finishFrame();
        zeno::log_debug("end frame {}", frame);
        if (chkfail()) return 1;
    }
    if (chkfail())
        return 1;
    zeno::log_info("program finished");
    return 0;
}

int offline_main(const char *zsgfile, int beginFrame, int endFrame);
int offline_main(const char *zsgfile, int beginFrame, int endFrame) {
    zeno::log_info("running in offline mode, file=[{}], begin={}, end={}", zsgfile, beginFrame, endFrame);

    GraphsManagment gman;
    gman.openZsgFile(zsgfile);
    IGraphsModel *pModel = gman.currentModel();
    ZASSERT_EXIT(pModel, 1);

	rapidjson::StringBuffer s;
	RAPIDJSON_WRITER writer(s);
    {
        JsonArrayBatch batch(writer);
        JsonHelper::AddVariantList({"setBeginFrameNumber", beginFrame}, "int", writer);
        JsonHelper::AddVariantList({"setEndFrameNumber", endFrame}, "int", writer);
        serializeScene(pModel, writer);
    }
    return offline_start(s.GetString());
}
