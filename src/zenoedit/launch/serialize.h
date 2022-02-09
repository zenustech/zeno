#ifndef __SERIALIZE_H__
#define __SERIALIZE_H__

#include <QtWidgets>

class GraphsModel;
class SubGraphModel;

void serializeScene(GraphsModel* pModel, QJsonArray& ret);

#endif
