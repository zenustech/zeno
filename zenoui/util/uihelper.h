#ifndef __UI_HELPER_H__
#define __UI_HELPER_H__

#include "../model/modeldata.h"
#include <rapidjson/document.h>

class UiHelper
{
public:
    static NODE_DESCS parseDescs(const rapidjson::Value &descs);
    static NODE_DESCS loadDescsFromTempFile();

private:
    static QVariant _parseDefaultValue(const QString &defaultValue);
    static PARAM_CONTROL _getControlType(const QString &type);
};

#endif