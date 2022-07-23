#ifndef __SERIALIZE_H__
#define __SERIALIZE_H__

#include <QtWidgets>
#include <QString>
#include <zenoui/util/jsonhelper.h>

class IGraphsModel;

void serializeScene(IGraphsModel* pModel, RAPIDJSON_WRITER& writer);
QString serializeSceneCpp(IGraphsModel* pModel);

#endif
