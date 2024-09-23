#include <zeno/core/CoreParam.h>
#include <zeno/core/INode.h>
#include <zeno/utils/helper.h>

namespace zeno {

    ParamObject ObjectParam::exportParam() const {
        ParamObject objparam;
        objparam.name = name;
        objparam.bInput = bInput;
        objparam.type = type;
        objparam.socketType = socketType;
        objparam.bVisible = bVisible;
        objparam.bEnable = bEnable;
        for (auto spLink : links) {
            objparam.links.push_back(getEdgeInfo(spLink));
        }
        objparam.wildCardGroup = wildCardGroup;
        return objparam;
    }

    ParamPrimitive PrimitiveParam::exportParam() const {
        ParamPrimitive param;
        param.name = name;
        param.bInput = bInput;
        param.type = type;
        param.socketType = socketType;
        param.defl = defl;
        param.result = result;
        param.control = control;
        param.ctrlProps = ctrlProps;
        param.bSocketVisible = bSocketVisible;
        param.wildCardGroup = wildCardGroup;
        param.sockProp = sockprop;
        param.bEnable = bEnable;
        param.bVisible = bVisible;
        for (auto spLink : links) {
            param.links.push_back(getEdgeInfo(spLink));
        }
        return param;
    }

}