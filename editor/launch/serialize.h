#ifndef __SERIALIZE_H__
#define __SERIALIZE_H__

#include <QtWidgets>

class GraphsModel;
class SubGraphModel;

QList<QStringList> serializeScene(GraphsModel* pModel);
QList<QStringList> serializeGraphs(GraphsModel* pModel);
QList<QStringList> serializeGraph(SubGraphModel* pModel);

#endif