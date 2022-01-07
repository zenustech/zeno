#include "serialize.h"
#include <model/graphsmodel.h>
#include <model/modeldata.h>
#include <model/modelrole.h>


QList<QStringList> serializeScene(GraphsModel* pModel)
{
    QList<QStringList> ret;
    ret.push_back(QStringList("clearAllState"));

    for (int i = 0; i < pModel->rowCount(); i++)
    {
        SubGraphModel* pSubModel = pModel->subGraph(i);
        const QString& name = pSubModel->name();
        ret.push_back({"switchGraph", name});
        const NODES_DATA& nodes = pSubModel->nodes();

        for (const NODE_DATA& node : nodes)
        {
            QString name = node[ROLE_OBJNAME].toString();
            const INPUT_SOCKETS& inputs = node[ROLE_INPUTS].value<INPUT_SOCKETS>();
            const PARAMS_INFO& params = node[ROLE_PARAMETERS].value<PARAMS_INFO>();
        }
    }

    return ret;
}

QList<QStringList> serializeGraphs(GraphsModel* pModel)
{
    QList<QStringList> ret;
    return ret;
}

QList<QStringList> serializeGraph(SubGraphModel* pModel)
{
    QList<QStringList> ret;
    return ret;
}