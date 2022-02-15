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

namespace {

struct ProgramRunData {
    inline static std::mutex g_mtx;

    std::string progJson;
    int nframes;
    //std::set<std::string> applies;

    void operator()() const {
        std::unique_lock _(g_mtx);

        zeno::log_info("launching program JSON: {}", progJson);

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
    QString strJson(doc.toJson(QJsonDocument::Compact));
    std::string progJson = strJson.toStdString();
    //QByteArray bytes = strJson.toUtf8();
    //std::string progJson = bytes.data();

    /*SubGraphModel* pMain = pModel->subGraph("main");
    NODES_DATA nodes = pMain->nodes();
    std::set<std::string> applies;
    for (NODE_DATA node : nodes)
    {
        int options = node[ROLE_OPTIONS].toInt();
        if (options & OPT_VIEW)
            applies.insert(node[ROLE_OBJID].toString().toStdString());
    }*/

    std::thread thr(ProgramRunData{std::move(progJson), nframes});//, std::move(applies)});
    thr.detach();
}

    /*for (int i = 0; i < nframes; i++)
    {
        // BEGIN XINXIN HAPPY >>>>>
        NODES_DATA nodes = pMain->nodes();
        for (NODE_DATA node : nodes)
        {
            //todo: special
            QString ident = node[ROLE_OBJID].toString();
            QString name = node[ROLE_OBJNAME].toString();
            PARAMS_INFO params = node[ROLE_PARAMETERS].value<PARAMS_INFO>();
            for (PARAM_INFO param : params)
            {
                if (param.value.type() == QVariant::Type::String)
                {
                    QString value = param.value.toString();
                    zeno::setNodeParam(ident.toStdString(), param.name.toStdString(), value.toStdString());
                }
            }
        }
        // ENDOF XINXIN HAPPY >>>>>
    }*/
