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
#include <future>
#include <zeno/core/data.h>


namespace zeno {

    class NodeImpl;
    class ObjectParam;
    class PrimitiveParam;
    class CoreParam;

    struct ObjectLink {
        std::future<zany> upstream_task;

        ObjectParam* fromparam = nullptr;  //IParam stored as unique ptr in the INode, so we need no smart pointer.
        ObjectParam* toparam = nullptr;
        std::string fromkey;    //for dict/list 对于list来说，keyName好像不合适，不过ILink本来就存在于links里面，已经是列表储存了。
        std::string tokey;
        std::string targetParam;
        bool bTraceAndTaken = false;
    };

    struct PrimitiveLink {
        std::future<zeno::reflect::Any> upstream_task;

        PrimitiveParam* fromparam = nullptr;
        PrimitiveParam* toparam = nullptr;
        std::string fromkey;    //for dict/list 对于list来说，keyName好像不合适，不过ILink本来就存在于links里面，已经是列表储存了。
        std::string tokey;
        std::string targetParam;
        bool bTraceAndTaken = false;
    };

    //引用连接，连接的双方都是基础类型的输入参数
    //目前是允许跨图层级的，也就是两个参数对应的节点不在同一个图里
    struct ReferLink {
        CoreParam* source_inparam = nullptr;
        PrimitiveParam* dest_inparam = nullptr;
    };


    struct CoreParam {
        std::string name;
        NodeImpl* m_wpNode;
        std::string wildCardGroup;
        std::string constrain;
        std::list<std::shared_ptr<ReferLink>> reflinks;

        ParamType type = Param_Null;
        SocketType socketType = NoSocket;
        bool bInput = true;
        bool m_idModify = false;    //该output param输出的obj是新创建的(false)还是基于已有的修改(true)
        bool bEnable = true;        //参数是否可用
        bool bVisible = true;       //参数是否可见，并非只指Socket可见，不过对于对象参数来说也就是socket
        bool bWildcard = false;     //是否为wildcard参数
    };

    struct ObjectParam : CoreParam {
        std::list<std::shared_ptr<ObjectLink>> links;
        zany spObject;

        ParamObject exportParam() const;
    };

    struct PrimitiveParam : CoreParam {
        zeno::reflect::Any defl;
        zeno::reflect::Any result;
        std::list<std::shared_ptr<PrimitiveLink>> links;
        ParamControl control = NullControl;
        zeno::reflect::Any ctrlProps;
        zeno::SocketProperty sockprop = zeno::Socket_Normal;
        bool bSocketVisible = true;
        bool bInnerParam = false;

        ParamPrimitive exportParam() const;
    };

}