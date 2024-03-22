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

    using SharedObjects = std::map<std::string, std::shared_ptr<zeno::IObject>>;

    struct RenderObjsInfo {
        SharedObjects newObjs;
        SharedObjects modifyObjs;
        std::set<std::string> remObjs;
        SharedObjects lightObjs;    //TODO:
        SharedObjects allObjects;

        bool empty() const {
            return newObjs.empty() && modifyObjs.empty() && remObjs.empty() && lightObjs.empty();
        }
    };


class ObjectManager
{
    struct _ObjInfo {
        std::shared_ptr<IObject> obj;
        std::set<ObjPath> attach_nodes;
        //ObjPath view_node;
    };

    using ViewObjects = std::map<std::string, _ObjInfo>;
    

    enum CacheType {
        MemoryCache,
        DiskCache
    };

    struct FrameData {
        std::optional<ViewObjects> view_objs;
        FILE* m_file = nullptr;
        CacheType cache;
    };

public:
     ObjectManager();
    ~ObjectManager();

    ZENO_API void beforeRun();
    ZENO_API void afterRun();

    ZENO_API void collectingObject(const std::string& id, std::shared_ptr<IObject> obj, std::shared_ptr<INode> view_node, bool bView);
    CALLBACK_REGIST(collectingObject, void, std::shared_ptr<IObject>, bool)

    ZENO_API void removeObject(const std::string& id);
    CALLBACK_REGIST(removeObject, void, std::string)

    ZENO_API void notifyTransfer(std::shared_ptr<IObject> obj);
    CALLBACK_REGIST(notifyTransfer, void, std::shared_ptr<IObject>)

    ZENO_API void viewObject(std::shared_ptr<IObject> obj, bool bView);
    CALLBACK_REGIST(viewObject, void, std::shared_ptr<IObject>, bool)

    ZENO_API int registerObjId(const std::string& objprefix);

    ZENO_API std::set<ObjPath> getAttachNodes(const std::string& id);

    ZENO_API void commit();
    ZENO_API void revert();

    ZENO_API void export_loading_objs(RenderObjsInfo& info);
    ZENO_API void export_all_view_objs(RenderObjsInfo& info);
    ZENO_API void export_all_view_objs(std::vector<std::pair<std::string, std::shared_ptr<zeno::IObject>>>& info);
    ZENO_API std::shared_ptr<IObject> getObj(std::string name);
    ZENO_API void clear_last_run();
    ZENO_API void collect_removing_objs(const std::string& objkey);
    ZENO_API void remove_attach_node_by_removing_objs();

    //viewport interactive obj
    ZENO_API void markObjInteractive(std::set<std::string>& newobjKeys);
    ZENO_API void unmarkObjInteractive(std::set<std::string>& removeobjKeys);
    ZENO_API void getModifyObjsInfo(std::map<std::string, std::shared_ptr<zeno::IObject>>& modifyInteractiveObjs);  //interactive objs

private:
    void clear();

    std::map<std::string, int> m_objRegister;

    ViewObjects m_objects;  //记录所有当前计算的对象，当切换帧的时候，可能使得部分依赖帧的对象重算。
    std::map<int, FrameData> m_frameData;   //记录流体相关的帧缓存

    std::set<std::string> m_viewObjs;
    std::set<std::string> m_lastViewObjs;
    std::set<std::string> m_removing_objs;  //这里是删除节点时记录的要删除的obj，要考虑rollback的情况

    std::set<std::string> m_newAdded;
    std::set<std::string> m_remove;

    std::set<std::string> m_modify;         //viewport interactive obj

    mutable std::mutex m_mtx;
};

}