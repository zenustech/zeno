#pragma once

#include <zeno/utils/api.h>
#include <zeno/core/Descriptor.h>
#include <zeno/core/data.h>
#include <zeno/core/INode.h>
#include <memory>
#include <string>
#include <atomic>
#include <map>
#include <zeno/core/common.h>
#include <zeno/io/iocommon.h>


namespace zeno {

    namespace reflect {
        class TypeBase;
    }

struct Graph;
struct Session;
struct NodeImpl;
struct GlobalVariableManager;
struct ObjectRecorder;
struct IObject;
struct INodeClass;
struct GlobalState;
struct GlobalComm;
struct GlobalError;
struct EventCallbacks;
struct UserData;
struct CalcManager;
struct AssetsMgr;
struct PythonEnvWrapper;
class FunctionManager;

struct Session {
    std::map<std::string, std::unique_ptr<INodeClass>> nodeClasses;

    std::unique_ptr<GlobalState> const globalState;
    std::unique_ptr<GlobalComm> const globalComm;
    std::unique_ptr<GlobalError> const globalError;
    std::unique_ptr<EventCallbacks> const eventCallbacks;
    std::unique_ptr<UserData> const m_userData;
    std::shared_ptr<Graph> mainGraph;
    std::unique_ptr<AssetsMgr> assets;
    std::unique_ptr<GlobalVariableManager> globalVariableManager;
    std::unique_ptr<FunctionManager> funcManager;
    std::unique_ptr<ObjectRecorder> m_recorder;

    ZENO_API Session();
    ZENO_API ~Session();
    ZENO_API void destroy();

    Session(Session const &) = delete;
    Session &operator=(Session const &) = delete;
    Session(Session &&) = delete;
    Session &operator=(Session &&) = delete;

    ZENO_API UserData &userData() const;
    ZENO_API std::shared_ptr<Graph> createGraph(const std::string& name);
    ZENO_API NodeImpl* getNodeByUuidPath(std::string const& uuid_path);
    ZENO_API NodeImpl* getNodeByPath(std::string const& uuid_path);
    ZENO_API void resetMainGraph();
    ZENO_API void clearMainGraph();
    ZENO_API void clearAssets();
    ZENO_API bool run(const std::string& currgraph, render_reload_info& infos);
    ZENO_API void interrupt();
    ZENO_API bool is_interrupted() const { return m_bInterrupted; }
    ZENO_API bool is_async_executing() const;
    ZENO_API void set_async_executing(bool bOn);
    ZENO_API unsigned long mainThreadId() const;
    ZENO_API void setMainThreadId(unsigned long threadId);
    ZENO_API void set_solver(const std::string& solver);
    ZENO_API std::string get_solver();
    ZENO_API void terminate_solve();
    ZENO_API void init_project_path(const std::wstring& path);
    ZENO_API std::wstring get_project_path() const;
    //ZENO_API 
    ZENO_API void set_auto_run(bool bOn);
    ZENO_API bool is_auto_run() const;
    ZENO_API bool is_frame_node(const std::string& node_cls);
    ZENO_API void markDirtyAndCleanResult();
    ZENO_API std::string dumpDescriptorsJSON() const;
    ZENO_API zeno::NodeRegistry dumpCoreCates();
    ZENO_API void defNodeClass(INode*(*ctor)(), std::string const &id, Descriptor const &desc = {});
    ZENO_API void defNodeClass2(INode*(*ctor)(), std::string const& nodecls, CustomUI const& customui);
    ZENO_API void defNodeClass3(INode*(*ctor)(), const char* pName, Descriptor const& desc = {});
    ZENO_API zeno::CustomUI getOfficalUIDesc(const std::string& clsname, bool& bExist);
    //ZENO_API void defNodeReflectClass(std::function<INode*()> ctor, zeno::reflect::TypeBase* pTypeBase);
    ZENO_API void beginLoadModule(const std::string& module_name);
    ZENO_API void uninstallModule(const std::string& module_name);
    ZENO_API void endLoadModule();
    ZENO_API void setApiLevelEnable(bool bEnable);
    ZENO_API void beginApiCall();
    ZENO_API void endApiCall();
    ZENO_API void setDisableRunning(bool bOn);
    ZENO_API void switchToFrame(int frameid);
    ZENO_API void updateFrameRange(int start, int end);
    ZENO_API void registerRunTrigger(std::function<void()> func);
    ZENO_API void registerNodeCallback(F_NodeStatus func);
    ZENO_API void registerIOCallback(F_IOProgress func);
    ZENO_API std::shared_ptr<Graph> getGraphByPath(const std::string& path);
    ZENO_API void registerObjUIInfo(size_t hashcode, std::string_view color, std::string_view nametip);
    ZENO_API bool getObjUIInfo(size_t hashcode, std::string_view& color, std::string_view& nametip);
    ZENO_API void initEnv(const zenoio::ZSG_PARSE_RESULT ioresult);
    ZENO_API void initPyzen(std::function<void()> pyzenFunc);
    ZENO_API void asyncRunPython(const std::string& code);
    ZENO_API void* hEventOfPyFinish();
    void reportNodeStatus(const ObjPath& path, bool bDirty, NodeRunStatus status);
    void reportIOProgress(const std::string& info, int inc);

private:
    int m_apiLevel = 0;
    bool m_bApiLevelEnable = true;
    bool m_bAutoRun = false;
    bool m_bDisableRunning = false;
    bool m_bReentrance = false;
    bool m_bAsyncExecute = false;
    std::string m_current_loading_module;
    std::string m_solver;
    std::wstring m_proj_path;
    unsigned long m_mainThreadId;
    std::atomic<bool> m_bInterrupted;

#ifdef ZENO_WITH_PYTHON
    std::unique_ptr<PythonEnvWrapper> m_pyWrapper;
#endif

    zeno::NodeRegistry m_cates;
    //std::map<std::string, std::vector<NodeInfo>> m_cates;
    //std::map<std::string, std::vector<std::string>> m_module_nodes;

    std::function<void()> m_callbackRunTrigger;
    F_NodeStatus m_funcNodeStatus;
    F_IOProgress m_funcIOCallback;
};

ZENO_API Session &getSession();

}
