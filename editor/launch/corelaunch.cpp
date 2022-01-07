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

        QList<QStringList> ret = serializeScene(pModel);

        std::string strs = zeno::dumpDescriptors();
    }
}

void cleanIOPath()
{

}