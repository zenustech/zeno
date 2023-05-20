#ifndef __SERIALIZE_H__
#define __SERIALIZE_H__

#include <QtWidgets>
#include <QString>
#include <zenomodel/include/jsonhelper.h>

class IGraphsModel;

void serializeScene(IGraphsModel* pModel, RAPIDJSON_WRITER& writer, bool applyLightAndCameraOnly = false, bool applyMaterialOnly = false);
QString serializeSceneCpp(IGraphsModel* pModel);

#endif
