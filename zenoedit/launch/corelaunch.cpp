#include "corelaunch.h"
#include <model/graphsmodel.h>
#include <zenoui/io/zsgwriter.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/zeno.h>
#include "serialize.h"

namespace {

struct ProgramRunData {
    std::string progJson;
    int nframes;
    std::set<std::string> applies;

    void run() {
        zeno::loadScene(progJson.c_str());

        zeno::switchGraph("main");

        for (int i = 0; i < nframes; i++) {
            zeno::state.frameBegin();
            while (zeno::state.substepBegin())
            {
                zeno::applyNodes(applies);
                zeno::state.substepEnd();
            }
            //zeno::Visualization::endFrame();
            zeno::state.frameEnd();
        }
    }
};

}

static ProgramRunData g_program_run_data;

static void program_runner() {
    g_program_run_data.run();
}

void launchProgram(GraphsModel* pModel, int nframes)
{
    QJsonArray ret;
    serializeScene(pModel, ret);

    QJsonDocument doc(ret);
    QString strJson(doc.toJson(QJsonDocument::Compact));
    QByteArray bytes = strJson.toUtf8();
    std::string progJson = bytes.data();

    SubGraphModel* pMain = pModel->subGraph("main");
    NODES_DATA nodes = pMain->nodes();
    std::set<std::string> applies;
    for (NODE_DATA node : nodes)
    {
        int options = node[ROLE_OPTIONS].toInt();
        if (options & OPT_VIEW)
            applies.insert(node[ROLE_OBJID].toString().toStdString());
    }

    static std::thread runner_thread(program_runner);
    g_program_run_data = {std::move(progJson), nframes, std::move(applies)};

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
}
