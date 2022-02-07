#include "corelaunch.h"
#include <model/graphsmodel.h>
#include <zenoui/io/zsgwriter.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/zeno.h>
#include "serialize.h"
#include <thread>

namespace {

struct ProgramRunData {
    inline static bool g_running = false;

    std::string progJson;
    int nframes;
    std::set<std::string> applies;

    void operator()() {
        zeno::loadScene(progJson.c_str());

        zeno::switchGraph("main");

        for (int i = 0; i < nframes; i++) {
            zeno::getSession().globalState->frameBegin();
            while (zeno::getSession().globalState->substepBegin())
            {
                zeno::applyNodes(applies);
                zeno::getSession().globalState->substepEnd();
            }
            //zeno::Visualization::endFrame();
            zeno::getSession().globalState->frameEnd();
        }

        g_running = false;
    }
};

}

void launchProgram(GraphsModel* pModel, int nframes)
{
    if (ProgramRunData::g_running) {
        printf("A program is already running! Please kill first\n");
        return;
    }

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

    std::thread thr(ProgramRunData{std::move(progJson), nframes, std::move(applies)});
    ProgramRunData::g_running = true;
    thr.detach();

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
