#ifndef __SERIALIZE_H__
#define __SERIALIZE_H__

#include <QtWidgets>
#include <zenoui/util/jsonhelper.h>

class GraphsModel;
class SubGraphModel;

void serializeScene(GraphsModel* pModel, RAPIDJSON_WRITER& writer);
QString translateGraphToCpp(const char *subgJson, size_t subgJsonLen, GraphsModel *model);

#endif
