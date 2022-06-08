#ifndef __SERIALIZE_H__
#define __SERIALIZE_H__

#include <QtWidgets>
#include <zenoui/util/jsonhelper.h>

class IGraphsModel;

void serializeScene(IGraphsModel* pModel, RAPIDJSON_WRITER& writer);
QString translateGraphToCpp(const char *subgJson, size_t subgJsonLen, IGraphsModel *model);

#endif
