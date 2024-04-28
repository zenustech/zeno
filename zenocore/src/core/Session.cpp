#include <zeno/core/Session.h>
#include <zeno/core/IObject.h>
#include <zeno/core/Assets.h>
#include <zeno/core/ObjectManager.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/extra/GlobalComm.h>
#include <zeno/extra/GlobalError.h>
#include <zeno/extra/EventCallbacks.h>
#include <zeno/types/UserData.h>
#include <zeno/core/Graph.h>
#include <zeno/core/INode.h>
#include <zeno/core/IParam.h>
#include <zeno/utils/safe_at.h>
#include <zeno/utils/logger.h>
#include <zeno/utils/string.h>
#include <zeno/utils/helper.h>
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include <zeno/extra/SubnetNode.h>
#include <zeno/extra/GraphException.h>

namespace zeno {

namespace {

struct ImplNodeClass : INodeClass {
    std::shared_ptr<INode>(*ctor)();

    ImplNodeClass(std::shared_ptr<INode>(*ctor)(), CustomUI const& customui, std::string const& name)
        : INodeClass(customui, name), ctor(ctor) {}

    virtual std::shared_ptr<INode> new_instance(std::shared_ptr<Graph> pGraph, std::string const &name) const override {
        std::shared_ptr<INode> spNode = ctor();
        spNode->initUuid(pGraph, classname);

        std::shared_ptr<SubnetNode> spSubnet = std::dynamic_pointer_cast<SubnetNode>(spNode);

        spNode->set_name(name);

        //init all params, and set defl value
        for (const ParamTab& tab : m_customui.tabs)
        {
            for (const ParamGroup& group : tab.groups)
            {
                for (const ParamInfo& param : group.params)
                {
                    std::shared_ptr<IParam> sparam = std::make_shared<IParam>();
                    sparam->name = param.name;
                    sparam->m_wpNode = spNode;
                    sparam->type = param.type;
                    sparam->defl = param.defl;
                    sparam->socketType = param.socketType;
                    sparam->control = param.control;
                    sparam->optCtrlprops = param.ctrlProps;
                    spNode->add_input_param(sparam);
                }
            }
        }

        for (const ParamInfo& param : m_customui.outputs)
        {
            std::shared_ptr<IParam> sparam = std::make_shared<IParam>();
            sparam->name = param.name;
            sparam->m_wpNode = spNode;
            sparam->type = param.type;
            sparam->socketType = PrimarySocket;
            spNode->add_output_param(sparam);
        }

        return spNode;
    }
};

}

ZENO_API Session::Session()
    : globalState(std::make_unique<GlobalState>())
    , globalComm(std::make_unique<GlobalComm>())
    , globalError(std::make_unique<GlobalError>())
    , eventCallbacks(std::make_unique<EventCallbacks>())
    , m_userData(std::make_unique<UserData>())
    , mainGraph(std::make_shared<Graph>("main"))
    , assets(std::make_shared<AssetsMgr>())
    , objsMan(std::make_unique<ObjectManager>())
{
    initNodeCates();
}

ZENO_API Session::~Session() = default;

ZENO_API void Session::defNodeClass(std::shared_ptr<INode>(*ctor)(), std::string const &clsname, Descriptor const &desc) {
    if (nodeClasses.find(clsname) != nodeClasses.end()) {
        log_error("node class redefined: `{}`\n", clsname);
    }
    CustomUI ui = descToCustomui(desc);
    auto cls = std::make_unique<ImplNodeClass>(ctor, ui, clsname);
    nodeClasses.emplace(clsname, std::move(cls));
}

ZENO_API void Session::defNodeClass2(std::shared_ptr<INode>(*ctor)(), std::string const& nodecls, CustomUI const& customui) {
    if (nodeClasses.find(nodecls) != nodeClasses.end()) {
        log_error("node class redefined: `{}`\n", nodecls);
    }
    CustomUI ui = customui;
    initControlsByType(ui);
    auto cls = std::make_unique<ImplNodeClass>(ctor, ui, nodecls);
    nodeClasses.emplace(nodecls, std::move(cls));
}


ZENO_API INodeClass::INodeClass(CustomUI const &customui, std::string const& classname)
    : m_customui(customui)
    , classname(classname)
{
}

ZENO_API INodeClass::~INodeClass() = default;

ZENO_API std::shared_ptr<Graph> Session::createGraph(const std::string& name) {
    auto graph = std::make_shared<Graph>(name);
    return graph;
}

ZENO_API void Session::resetMainGraph() {
    mainGraph.reset();
    mainGraph = std::make_shared<Graph>("main");
}

ZENO_API void Session::setApiLevelEnable(bool bEnable)
{
    m_bApiLevelEnable = bEnable;
}

ZENO_API void Session::beginApiCall()
{
    if (!m_bApiLevelEnable) return;
    m_apiLevel++;
}

ZENO_API void Session::endApiCall()
{
    if (!m_bApiLevelEnable) return;
    m_apiLevel--;
    if (m_apiLevel == 0) {
        if (m_bAutoRun) {
            if (m_callbackRunTrigger) {
                m_callbackRunTrigger();
            }
            else {
                run();
            }
        }
    }
}

ZENO_API void Session::registerRunTrigger(std::function<void()> func)
{
    m_callbackRunTrigger = func;
}

ZENO_API void Session::registerNodeCallback(F_NodeStatus func)
{
    m_funcNodeStatus = func;
}

void Session::reportNodeStatus(const ObjPath& path, bool bDirty, NodeRunStatus status)
{
    if (m_funcNodeStatus) {
        m_funcNodeStatus(path, bDirty, status);
    }
}

ZENO_API int Session::registerObjId(const std::string& objprefix)
{
    int objid = objsMan->registerObjId(objprefix);
    return objid;
}

ZENO_API void Session::switchToFrame(int frameid)
{
    CORE_API_BATCH
    mainGraph->markDirtyWhenFrameChanged();
    globalState->updateFrameId(frameid);
}

ZENO_API void Session::interrupt() {
    m_bInterrupted = true;
}

ZENO_API bool Session::is_interrupted() const {
    return m_bInterrupted;
}

ZENO_API bool Session::run() {
    m_bInterrupted = false;
    globalState->set_working(true);

    zeno::log_info("Session::run()");

    objsMan->beforeRun();
    zeno::scope_exit sp([&]() { objsMan->afterRun(); });

    globalError->clearState();

    //本次运行清除m_objects中上一次运行时被标记移除的obj，不能立刻清除因为视窗export_loading_objs时，需要在m_objects中查找被删除的obj
    objsMan->clearLastUnregisterObjs();
    //对之前删除节点时记录的obj，对应的所有其他关联节点，都标脏
    objsMan->remove_attach_node_by_removing_objs();

    zeno::GraphException::catched([&] {
        mainGraph->runGraph();
        }, *globalError);
    if (globalError->failed()) {
        zeno::log_error("");
    }

    return true;
}

ZENO_API void Session::set_auto_run(bool bOn) {
    m_bAutoRun = bOn;
}

ZENO_API bool Session::is_auto_run() const {
    return m_bAutoRun;
}

ZENO_API void Session::set_Rerun()
{
    mainGraph->markDirtyAll();
    objsMan->clear();
}

void Session::initNodeCates() {
    for (auto const& [key, cls] : nodeClasses) {
        if (!key.empty() && key.front() == '^')
            continue;
        std::string cate = cls->m_customui.category;
        if (m_cates.find(cate) == m_cates.end())
            m_cates.insert(std::make_pair(cate, std::vector<std::string>()));
        m_cates[cate].push_back(key);
    }
}

ZENO_API zeno::NodeCates Session::dumpCoreCates() {
    if (m_cates.empty()) {
        initNodeCates();
    }
    return m_cates;
}

namespace {
std::string dumpDescriptorToJson(const std::string &key, const Descriptor& descriptor) {
    using namespace rapidjson;
    Document doc;
    doc.SetArray();

    // Inputs array
    Value inputs(kArrayType);
    for (const auto& input : descriptor.inputs) {
        Value inputArray(kArrayType);
        inputArray.PushBack(Value().SetString(input.type.c_str(), doc.GetAllocator()), doc.GetAllocator());
        inputArray.PushBack(Value().SetString(input.name.c_str(), doc.GetAllocator()), doc.GetAllocator());
        inputArray.PushBack(Value().SetString(input.defl.c_str(), doc.GetAllocator()), doc.GetAllocator());
        inputArray.PushBack(Value().SetString(input.doc.c_str(), doc.GetAllocator()), doc.GetAllocator());
        inputs.PushBack(inputArray, doc.GetAllocator());
    }

    // Outputs array
    Value outputs(kArrayType);
    for (const auto& output : descriptor.outputs) {
        Value outputArray(kArrayType);
        outputArray.PushBack(Value().SetString(output.type.c_str(), doc.GetAllocator()), doc.GetAllocator());
        outputArray.PushBack(Value().SetString(output.name.c_str(), doc.GetAllocator()), doc.GetAllocator());
        outputArray.PushBack(Value().SetString(output.defl.c_str(), doc.GetAllocator()), doc.GetAllocator());
        outputArray.PushBack(Value().SetString(output.doc.c_str(), doc.GetAllocator()), doc.GetAllocator());
        outputs.PushBack(outputArray, doc.GetAllocator());
    }

    // Params array
    Value params(kArrayType);
    for (const auto& param : descriptor.params) {
        Value paramArray(kArrayType);
        paramArray.PushBack(Value().SetString(param.type.c_str(), doc.GetAllocator()), doc.GetAllocator());
        paramArray.PushBack(Value().SetString(param.name.c_str(), doc.GetAllocator()), doc.GetAllocator());
        paramArray.PushBack(Value().SetString(param.defl.c_str(), doc.GetAllocator()), doc.GetAllocator());
        paramArray.PushBack(Value().SetString(param.doc.c_str(), doc.GetAllocator()), doc.GetAllocator());
        params.PushBack(paramArray, doc.GetAllocator());
    }

    // Categories array
    Value categories(kArrayType);
    for (const auto& category : descriptor.categories) {
        categories.PushBack(Value().SetString(category.c_str(), doc.GetAllocator()), doc.GetAllocator());
    }

    // Push values into the main document
    doc.PushBack(Value().SetString(key.c_str(), doc.GetAllocator()), doc.GetAllocator());
    doc.PushBack(inputs, doc.GetAllocator());
    doc.PushBack(outputs, doc.GetAllocator());
    doc.PushBack(params, doc.GetAllocator());
    doc.PushBack(categories, doc.GetAllocator());
    doc.PushBack(Value().SetString(descriptor.doc.c_str(), doc.GetAllocator()), doc.GetAllocator());

    // Write the JSON string to stdout
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    doc.Accept(writer);
    return buffer.GetString();
}
}

ZENO_API std::string Session::dumpDescriptorsJSON() const {
    //deprecated.
    return "";
}

ZENO_API UserData &Session::userData() const {
    return *m_userData;
}

ZENO_API Session &getSession() {
#if 0
    static std::unique_ptr<Session> ptr;
    if (!ptr) {
        ptr = std::make_unique<Session>();
    }
#else
    static std::unique_ptr<Session> ptr = std::make_unique<Session>();
#endif
    return *ptr;
}

}
