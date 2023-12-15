#ifndef __SERIALIZE_H__
#define __SERIALIZE_H__

#include <QtWidgets>
#include <QString>
#include <zenomodel/include/jsonhelper.h>

class IGraphsModel;

void serializeScene(IGraphsModel* pModel, RAPIDJSON_WRITER& writer, const QString& configPath = "");
QString serializeSceneCpp(IGraphsModel* pModel);

#endif
