#ifndef __CUSTOMUI_READWRITE_H__
#define __CUSTOMUI_READWRITE_H__

#include <zenomodel/include/jsonhelper.h>

class ViewParamModel;

namespace zenoio
{
    void exportCustomUI(ViewParamModel* pModel, RAPIDJSON_WRITER& writer);
}

#endif