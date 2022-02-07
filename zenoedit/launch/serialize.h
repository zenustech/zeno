#ifndef __SERIALIZE_H__
#define __SERIALIZE_H__

#include <QtWidgets>

class GraphsModel;
class SubGraphModel;

void serializeScene(GraphsModel* pModel, QJsonArray& ret);
QJsonArray serializeGraphs(GraphsModel* pModel);
void serializeGraph(SubGraphModel* pModel, const QStringList& graphNames, QJsonArray& ret);

#endif