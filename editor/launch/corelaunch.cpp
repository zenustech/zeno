#include "corelaunch.h"
#include <model/graphsmodel.h>
#include <QTemporaryFile>
#include <io/zsgwriter.h>
#include <extra/GlobalState.h>
#include <extra/Visualization.h>
#include <zeno.h>
#include "serialize.h"
#ifdef Q_OS_LINUX
#include <dlfcn.h>
#endif

QString g_iopath;

void launchProgram(GraphsModel* pModel, int nframes)
{
    //todo
#ifdef Q_OS_WIN
    LoadLibrary("zeno_ZenoFX.dll");
    LoadLibrary("zeno_oldzenbase.dll");
#else
    void* dp = nullptr;
    dp = dlopen("libzeno_ZenoFX.so", RTLD_NOW);
    if (dp == nullptr)
        return;
    dp = dlopen("libzeno_oldzenbase.so", RTLD_NOW);
    if (dp == nullptr)
        return;
#endif

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

        zeno::state = zeno::GlobalState();
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

            zeno::state.frameBegin();
            while (zeno::state.substepBegin())
            {
                zeno::applyNodes(applies);
                zeno::state.substepEnd();
            }
#ifdef ZENO_VISUALIZATION
            zeno::Visualization::endFrame();
#endif
            zeno::state.frameEnd();
        }

        //print('EXITING')
    }
}

void cleanIOPath()
{

}
