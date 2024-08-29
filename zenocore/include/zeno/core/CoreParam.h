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
#include <zeno/core/data.h>


namespace zeno {

class INode;
class ObjectParam;
class PrimitiveParam;

struct ObjectLink {
    ObjectParam* fromparam = nullptr;  //IParam stored as unique ptr in the INode, so we need no smart pointer.
    ObjectParam* toparam = nullptr;
    std::string fromkey;    //for dict/list ����list��˵��keyName���񲻺��ʣ�����ILink�����ʹ�����links���棬�Ѿ����б������ˡ�
    std::string tokey;
    std::string targetParam;
};


struct PrimitiveLink {
    PrimitiveParam* fromparam = nullptr;
    PrimitiveParam* toparam = nullptr;
    std::string targetParam;
};

struct CoreParam {
    std::string name;
    std::weak_ptr<INode> m_wpNode;
    
    ParamType type = Param_Null;
    SocketType socketType = NoSocket;
    bool bInput = true;
    bool m_idModify = false;    //��output param�����obj���´�����(false)���ǻ������е��޸�(true)
    std::string wildCardGroup;
};

struct ObjectParam : CoreParam {
    std::list<std::shared_ptr<ObjectLink>> links;
    zany /*zeno::reflect::Any*/ spObject;        //ֻ�������ָ�룬��ʵ�Ѿ���һ��"any"�ˡ�

    ParamObject exportParam() const;
};

struct PrimitiveParam : CoreParam {
    zeno::reflect::Any defl;
    zeno::reflect::Any result;
    std::list<std::shared_ptr<PrimitiveLink>> links;
    ParamControl control = NullControl;
    zeno::reflect::Any ctrlProps;
    bool bVisible = true;

    ParamPrimitive exportParam() const;
};

}