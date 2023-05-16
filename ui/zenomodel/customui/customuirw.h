#ifndef __CUSTOMUI_READWRITE_H__
#define __CUSTOMUI_READWRITE_H__

#include <QObject>

#include <zenomodel/include/viewparammodel.h>
#include <zenomodel/include/jsonhelper.h>
#include <zenomodel/include/modeldata.h>

namespace zenomodel
{
    void exportItem(const VParamItem* pItem, RAPIDJSON_WRITER& writer);
    void exportCustomUI(ViewParamModel* pModel, RAPIDJSON_WRITER& writer);
    VPARAM_INFO importParam(const QString& paramName, const rapidjson::Value& paramVal);
    VPARAM_INFO importCustomUI(const rapidjson::Value& jsonCutomUI);
    
}

#endif