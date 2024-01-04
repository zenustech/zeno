#ifndef __SERIALIZE_H__
#define __SERIALIZE_H__

#include <QtWidgets>
#include <QString>
#include <zenomodel/include/jsonhelper.h>
#include "corelaunch.h"

class IGraphsModel;

void serializeScene(IGraphsModel* pModel, RAPIDJSON_WRITER& writer, LAUNCH_PARAM param);
QString serializeSceneCpp(IGraphsModel* pModel);

#endif
