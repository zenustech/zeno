#pragma once

#include <zeno/core/IObject.h>
#include <zeno/core/INode.h>
#include <zeno/utils/PolymorphicMap.h>
#include <zeno/utils/api.h>
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <map>
#include <set>
#include <functional>


namespace zeno {

    extern ZENO_API std::recursive_mutex g_objsMutex;

class ObjectManager
{
    struct _ObjInfo {
        std::shared_ptr<IObject> obj;
        std::set<ObjPath> attach_nodes;
        //ObjPath view_node;
    };

public:
     ObjectManager();
    ~ObjectManager();

    ZENO_API void addObject(const std::string& id, std::shared_ptr<IObject> obj, std::shared_ptr<INode> view_node, bool bView);
    CALLBACK_REGIST(addObject, void, std::shared_ptr<IObject>, bool)

    ZENO_API void removeObject(const std::string& id);
    CALLBACK_REGIST(removeObject, void, std::string)

    ZENO_API void notifyTransfer(std::shared_ptr<IObject> obj);
    CALLBACK_REGIST(notifyTransfer, void, std::shared_ptr<IObject>)

    ZENO_API void viewObject(std::shared_ptr<IObject> obj, bool bView);
    CALLBACK_REGIST(viewObject, void, std::shared_ptr<IObject>, bool)

    ZENO_API int registerObjId(const std::string& objprefix);

private:
    void clear();

    std::map<std::string, int> m_objRegister;
    std::map<std::string, _ObjInfo> m_objects;
    std::set<ObjPath> viewNodes;
    //std::set<std::string> m_viewObjs;
};

}