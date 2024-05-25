#pragma once

#include <zeno/utils/api.h>
#include <zeno/core/Descriptor.h>
#include <zeno/core/data.h>
#include <memory>
#include <string>
#include <map>
#include <zeno/core/common.h>

namespace zeno {

struct Graph;
struct Session;
struct INode;

struct INodeClass {
    CustomUI m_customui;
    std::string classname;

    ZENO_API INodeClass(CustomUI const &customui, std::string const& classname);
    ZENO_API virtual ~INodeClass();
    virtual std::shared_ptr<INode> new_instance(std::shared_ptr<Graph> pGraph, std::string const &classname) const = 0;
};

struct IObject;
struct GlobalState;
struct GlobalComm;
struct GlobalError;
struct EventCallbacks;
struct UserData;
struct CalcManager;
struct ObjectManager;
struct AssetsMgr;
struct ReferManager;

struct Session {
    std::map<std::string, std::unique_ptr<INodeClass>> nodeClasses;

    std::unique_ptr<GlobalState> const globalState;
    std::unique_ptr<GlobalComm> const globalComm;
    std::unique_ptr<GlobalError> const globalError;
    std::unique_ptr<EventCallbacks> const eventCallbacks;
    std::unique_ptr<UserData> const m_userData;
    std::unique_ptr<ObjectManager> const objsMan;
    std::shared_ptr<Graph> mainGraph;
    std::shared_ptr<AssetsMgr> assets;
    std::unique_ptr<ReferManager> referManager;

    ZENO_API Session();
    ZENO_API ~Session();

    Session(Session const &) = delete;
    Session &operator=(Session const &) = delete;
    Session(Session &&) = delete;
    Session &operator=(Session &&) = delete;

    ZENO_API UserData &userData() const;
    ZENO_API std::shared_ptr<Graph> createGraph(const std::string& name);
    ZENO_API void resetMainGraph();
    ZENO_API bool run();
    ZENO_API void interrupt();
    ZENO_API bool is_interrupted() const;
    //ZENO_API 
    ZENO_API void set_auto_run(bool bOn);
    ZENO_API bool is_auto_run() const;
    ZENO_API void set_Rerun();
    ZENO_API std::string dumpDescriptorsJSON() const;
    ZENO_API zeno::NodeCates dumpCoreCates();
    ZENO_API void defNodeClass(std::shared_ptr<INode>(*ctor)(), std::string const &id, Descriptor const &desc = {});
    ZENO_API void defNodeClass2(std::shared_ptr<INode>(*ctor)(), std::string const& nodecls, CustomUI const& customui);
    ZENO_API void setApiLevelEnable(bool bEnable);
    ZENO_API void beginApiCall();
    ZENO_API void endApiCall();
    ZENO_API void setDisableRunning(bool bOn);
    ZENO_API void switchToFrame(int frameid);
    ZENO_API int registerObjId(const std::string& objprefix);
    ZENO_API void registerRunTrigger(std::function<void()> func);
    ZENO_API void registerNodeCallback(F_NodeStatus func);
    void reportNodeStatus(const ObjPath& path, bool bDirty, NodeRunStatus status);

private:
    void initNodeCates();

    zeno::NodeCates m_cates;
    int m_apiLevel = 0;
    bool m_bApiLevelEnable = true;
    bool m_bAutoRun = false;
    bool m_bInterrupted = false;
    bool m_bDisableRunning = false;

    std::function<void()> m_callbackRunTrigger;
    F_NodeStatus m_funcNodeStatus;
};

ZENO_API Session &getSession();

}
