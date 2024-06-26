#pragma once

#include <zeno/utils/api.h>
#include <zeno/core/IObject.h>
//#include <zeno/core/INode.h>
#include <zeno/core/common.h>
#include <variant>
#include <memory>
#include <string>
#include <set>
#include <map>
#include <optional>

namespace zeno {

class INode;
class ObjectParam;
class PrimitiveParam;

struct ObjectLink {
    ObjectParam* fromparam = nullptr;  //IParam stored as unique ptr in the INode, so we need no smart pointer.
    ObjectParam* toparam = nullptr;
    std::string fromkey;    //for dict/list 对于list来说，keyName好像不合适，不过ILink本来就存在于links里面，已经是列表储存了。
    std::string tokey;
};


struct PrimitiveLink {
    PrimitiveParam* fromparam = nullptr;
    PrimitiveParam* toparam = nullptr;
};

struct CoreParam {
    std::string name;
    std::weak_ptr<INode> m_wpNode;
    
    ParamType type = Param_Null;
    SocketType socketType = NoSocket;
    bool bInput = true;
    bool m_idModify = false;    //该output param输出的obj是新创建的(false)还是基于已有的修改(true)
    std::string wildCardGroup;
};

struct ObjectParam : CoreParam {
    std::list<std::shared_ptr<ObjectLink>> links;
    zany spObject;

    ParamObject exportParam() const;
};

struct PrimitiveParam : CoreParam {
    zvariant defl;
    zvariant result;
    std::list<std::shared_ptr<PrimitiveLink>> links;
    ParamControl control = NullControl;
    std::optional<ControlProperty> optCtrlprops;
    bool bVisible = true;

    ParamPrimitive exportParam() const;
};

}