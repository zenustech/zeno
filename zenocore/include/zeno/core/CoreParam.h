#pragma once

#include <zeno/utils/api.h>
#include <zeno/core/IObject.h>
#include <zeno/core/common.h>
#include <variant>
#include <memory>
#include <string>
#include <set>
#include <map>
#include <optional>
#include <zeno/core/data.h>


namespace zeno {

class INode;
class ObjectParam;
class PrimitiveParam;

struct ObjectLink {
    ObjectParam* fromparam = nullptr;  //IParam stored as unique ptr in the INode, so we need no smart pointer.
    ObjectParam* toparam = nullptr;
    std::string fromkey;    //for dict/list 对于list来说，keyName好像不合适，不过ILink本来就存在于links里面，已经是列表储存了。
    std::string tokey;
    std::string targetParam;
};

struct PrimitiveLink {
    PrimitiveParam* fromparam = nullptr;
    PrimitiveParam* toparam = nullptr;
    std::string targetParam;
};

//引用连接，连接的双方都是基础类型的输入参数
struct ReferLink {
    PrimitiveParam* source_inparam = nullptr;
    PrimitiveParam* dest_inparam = nullptr;
};


struct CoreParam {
    std::string name;
    std::weak_ptr<INode> m_wpNode;
    std::string wildCardGroup;
    std::string constrain;

    ParamType type = Param_Null;
    SocketType socketType = NoSocket;
    bool bInput = true;
    bool m_idModify = false;    //该output param输出的obj是新创建的(false)还是基于已有的修改(true)
};

struct ObjectParam : CoreParam {
    std::list<std::shared_ptr<ObjectLink>> links;
    zany /*zeno::reflect::Any*/ spObject;        //只储存基类指针，其实已经是一种"any"了。

    ParamObject exportParam() const;
};

struct PrimitiveParam : CoreParam {
    zeno::reflect::Any defl;
    zeno::reflect::Any result;
    std::list<std::shared_ptr<PrimitiveLink>> links;
    std::list<std::shared_ptr<ReferLink>> reflinks;
    ParamControl control = NullControl;
    zeno::reflect::Any ctrlProps;
    zeno::SocketProperty sockprop = zeno::Socket_Normal;
    bool bVisible = true;
    bool bInnerParam = false;

    ParamPrimitive exportParam() const;
};

}