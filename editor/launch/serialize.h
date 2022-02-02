#ifndef __SERIALIZE_H__
#define __SERIALIZE_H__

#include <QtWidgets>

class GraphsModel;
class SubGraphModel;

void serializeScene(GraphsModel* pModel, QJsonArray& ret);
QJsonArray serializeGraphs(GraphsModel* pModel);
void serializeGraph(SubGraphModel* pModel, const QStringList& graphNames, QJsonArray& ret);

QJsonArray serializeScene(const QJsonObject& graphs);
QJsonArray serializeGraphs(const QJsonObject& graphs, bool has_subgraphs = true);
void serializeGraph(const QJsonObject& graphs, const QStringList& graphNames, QJsonArray& ret);

#endif