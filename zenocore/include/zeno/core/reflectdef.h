#pragma once

#include <memory>
#include "reflect/core.hpp"
#include "reflect/type"
#include "reflect/reflection_traits.hpp"


namespace zeno {

    struct _Param
    {
        std::string mapTo;
        std::string dispName;
        zeno::reflect::Any defl;
        std::string wildCardGroup;
        ParamControl ctrl;
        bool bInnerParam = false;
        zeno::reflect::Any ctrlProps;
    };

    struct _ObjectParam
    {
        std::string mapTo;
        std::string dispName;
        SocketType type;
        std::string wildCardGroup;
    };

    struct _ParamGroup {
        std::string name = "Group1";
        std::vector<_Param> params;
    };

    struct _ObjectGroup {
        std::vector<_ObjectParam> objs;
    };

    struct _ParamTab {
        std::string name = "Tab1";
        std::vector<_ParamGroup> groups;
    };

    struct ReflectCustomUI
    {
        _ObjectGroup inputObjs;
        _ObjectGroup outputObjs;
        _ObjectParam retInfo;       //也可以标识数值类型
        _ParamTab inputPrims;
        _ParamGroup outputPrims;
    };

}