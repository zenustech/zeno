#ifndef __CUSTOMUI_READWRITE_H__
#define __CUSTOMUI_READWRITE_H__

#include <QObject>

#include <zenomodel/include/jsonhelper.h>
#include <zenomodel/include/modeldata.h>

class ViewParamModel;

namespace zenoio
{
    VPARAM_INFO importCustomUI(const rapidjson::Value& jsonCutomUI);
    void exportCustomUI(ViewParamModel* pModel, RAPIDJSON_WRITER& writer);
}

#endif