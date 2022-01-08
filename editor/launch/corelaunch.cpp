#include "corelaunch.h"
#include <model/graphsmodel.h>
#include <QTemporaryFile>
#include <io/zsgwriter.h>
#include <extra/GlobalState.h>
#include <zeno.h>
#include "serialize.h"

QString g_iopath;

void launchProgram(GraphsModel* pModel, int nframes)
{
    //todo
    LoadLibrary("C:\\zeno2\\zenqt\\bin\\zeno_ZenoFX.dll");
    LoadLibrary("C:\\zeno2\\zenqt\\bin\\zeno_oldzenbase.dll");

    cleanIOPath();

    QTemporaryDir dir("zenvis-");
    dir.setAutoRemove(false);
    if (dir.isValid())
    {
        g_iopath = dir.path();
        //TODO: os.environ.get('ZEN_SPROC') or os.environ.get('ZEN_DOFORK')
        QString path = g_iopath + "/prog.zsg";
        QFile f(path);
        if (!f.open(QIODevice::WriteOnly)) {
            qWarning() << Q_FUNC_INFO << "Failed to open" << path << f.errorString();
            return;
        }
        const QString& strContent = ZsgWriter::getInstance().dumpProgramStr(pModel);
        f.write(strContent.toUtf8());
        f.close();

        QByteArray bytes = g_iopath.toLatin1();
        zeno::state.setIOPath(bytes.data());

        QJsonArray ret;
        serializeScene(pModel, ret);

		QJsonDocument doc(ret);
		QString strJson(doc.toJson(QJsonDocument::Compact));
        bytes = strJson.toUtf8();

        zeno::loadScene(bytes.data());

        SubGraphModel* pMain = pModel->subGraph("main");
        NODES_DATA nodes = pMain->nodes();
        std::set<std::string> applies;
        for (NODE_DATA node : nodes)
        {
            int options = node[ROLE_OPTIONS].toInt();
            if (options & OPT_VIEW)
                applies.insert(node[ROLE_OBJID].toString().toStdString());
        }

        zeno::switchGraph("main");

        for (int i = 0; i < nframes; i++)
        {
            //>>>
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
        }

        zeno::state.frameBegin();
        while (zeno::state.substepBegin())
        {
            zeno::applyNodes(applies);
            zeno::state.substepEnd();
        }
        zeno::state.frameEnd();
    }
}

void cleanIOPath()
{

}