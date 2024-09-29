#include <zeno/core/Session.h>
#include <zeno/core/IObject.h>
#include <zeno/core/INodeClass.h>
#include <zeno/core/Assets.h>
#include <zeno/core/ObjectManager.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/extra/GlobalComm.h>
#include <zeno/extra/GlobalError.h>
#include <zeno/extra/EventCallbacks.h>
#include <zeno/types/UserData.h>
#include <zeno/core/Graph.h>
#include <zeno/core/INode.h>
#include <zeno/core/CoreParam.h>
#include <zeno/utils/safe_at.h>
#include <zeno/utils/logger.h>
#include <zeno/utils/string.h>
#include <zeno/utils/helper.h>
#include <zeno/zeno.h>
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include <zeno/extra/SubnetNode.h>
#include <zeno/extra/GraphException.h>
#include <zeno/core/GlobalVariable.h>
#include <zeno/core/FunctionManager.h>
#include <regex>

#include <reflect/core.hpp>
#include <reflect/type.hpp>
#include <reflect/metadata.hpp>
#include <reflect/registry.hpp>
#include <reflect/container/object_proxy>
#include <reflect/container/any>
#include <reflect/container/arraylist>
#include <reflect/core.hpp>
#include <zeno/core/reflectdef.h>
#include "zeno_types/reflect/reflection.generated.hpp"
#include "zeno_nodes/reflect/reflection.generated.hpp"


using namespace zeno::reflect;
using namespace zeno::types;

namespace zeno {

    struct _ObjUIInfo
    {
        std::string_view name;
        std::string_view color;
    };
    static std::map<size_t, _ObjUIInfo> s_objsUIInfo;

ZENO_API Session::Session()
    : globalState(std::make_unique<GlobalState>())
    , globalComm(std::make_unique<GlobalComm>())
    , globalError(std::make_unique<GlobalError>())
    , eventCallbacks(std::make_unique<EventCallbacks>())
    , m_userData(std::make_unique<UserData>())
    , mainGraph(std::make_shared<Graph>("main"))
    , assets(std::make_shared<AssetsMgr>())
    , objsMan(std::make_unique<ObjectManager>())
    , globalVariableManager(std::make_unique<GlobalVariableManager>())
    , funcManager(std::make_unique<FunctionManager>())
{
    std::vector<zeno::reflect::Any> wtf;
    wtf.push_back(3);
    wtf.push_back(std::make_shared<PrimitiveObject>());
    wtf.push_back("abc");

    Any anyList = wtf;
    if (anyList.type().hash_code() == zeno::types::gParamType_AnyList) {
        std::vector<zeno::reflect::Any> lst = any_cast<std::vector<zeno::reflect::Any>>(anyList);
        for (auto& elem : lst) {
            int j;
            j = 0;
        }
    }
}

ZENO_API Session::~Session() = default;


static CustomUI descToCustomui(const Descriptor& desc) {
    //兼容以前写的各种ZENDEFINE
    CustomUI ui;

    ui.nickname = desc.displayName;
    ui.iconResPath = desc.iconResPath;
    ui.doc = desc.doc;
    if (!desc.categories.empty())
        ui.category = desc.categories[0];   //很多cate都只有一个

    ParamGroup default;
    for (const SocketDescriptor& param_desc : desc.inputs) {
        ParamType type = param_desc.type;
        if (isPrimitiveType(type)) {
            //如果是数值类型，就添加到组里
            ParamPrimitive param;
            param.name = param_desc.name;
            param.type = type;
            param.defl = zeno::str2any(param_desc.defl, param.type);
            convertToEditVar(param.defl, param.type);
            if (param_desc.socketType != zeno::NoSocket)
                param.socketType = param_desc.socketType;
            if (param_desc.control != NullControl)
                param.control = param_desc.control;
            if (!param_desc.comboxitems.empty()) {
                //compatible with old case of combobox items.
                param.type = zeno::types::gParamType_String;
                param.control = Combobox;
                std::vector<std::string> items = split_str(param_desc.comboxitems, ';');
                if (!items.empty()) {
                    items.erase(items.begin());
                    param.ctrlProps = items;
                }
            }
            if (param.type != Param_Null && param.control == NullControl)
                param.control = getDefaultControl(param.type);
            param.tooltip = param_desc.doc;
            param.sockProp = Socket_Normal;
            param.wildCardGroup = param_desc.wildCard;
            param.bSocketVisible = false;
            default.params.emplace_back(std::move(param));
        }
        else
        {
            //其他一律认为是对象（Zeno目前的类型管理非常混乱，有些类型值是空字符串，但绝大多数是对象类型
            ParamObject param;
            param.name = param_desc.name;
            param.type = type;
            if (param_desc.socketType != zeno::NoSocket)
                param.socketType = param_desc.socketType;
            param.bInput = true;
            param.wildCardGroup = param_desc.wildCard;

            //dict和list允许多连接口，且不限定对象类型（但只能是对象，暂不接收primitive，否则就违反了对象和primitive分开连的设计了）
            if (type == gParamType_Dict || type == gParamType_List) {
                param.sockProp = Socket_MultiInput;
            }
            else {
                param.sockProp = Socket_Normal;
            }

            ui.inputObjs.emplace_back(std::move(param));
        }
    }
    for (const ParamDescriptor& param_desc : desc.params) {
        ParamPrimitive param;
        param.name = param_desc.name;
        param.type = param_desc.type;
        param.defl = zeno::str2any(param_desc.defl, param.type);
        convertToEditVar(param.defl, param.type);
        param.socketType = NoSocket;
        //其他控件估计是根据类型推断的。
        if (!param_desc.comboxitems.empty()) {
            //compatible with old case of combobox items.
            param.type = zeno::types::gParamType_String;
            param.control = Combobox;
            std::vector<std::string> items = split_str(param_desc.comboxitems, ' ');
            if (!items.empty()) {
                items.erase(items.begin());
                param.ctrlProps = items;
            }
        }
        if (param.type != Param_Null && param.control == NullControl)
            param.control = getDefaultControl(param.type);
        param.tooltip = param_desc.doc;
        param.bSocketVisible = false;
        default.params.emplace_back(std::move(param));
    }
    for (const SocketDescriptor& param_desc : desc.outputs) {
        ParamType type = param_desc.type;
        if (isPrimitiveType(type)) {
            //如果是数值类型，就添加到组里
            ParamPrimitive param;
            param.name = param_desc.name;
            param.type = type;
            param.defl = zeno::str2any(param_desc.defl, param.type);
            //输出的数据端口没必要将vec转为vecedit
            if (param_desc.socketType != zeno::NoSocket)
                param.socketType = param_desc.socketType;
            param.control = NullControl;
            param.tooltip = param_desc.doc;
            param.sockProp = Socket_Normal;
            param.wildCardGroup = param_desc.wildCard;
            param.bSocketVisible = false;
            ui.outputPrims.emplace_back(std::move(param));
        }
        else
        {
            //其他一律认为是对象（Zeno目前的类型管理非常混乱，有些类型值是空字符串，但绝大多数是对象类型
            ParamObject param;
            param.name = param_desc.name;
            param.type = type;
            if (param_desc.socketType != zeno::NoSocket)
                param.socketType = param_desc.socketType;
            if (param.socketType != zeno::Socket_WildCard)  //输出可能是wildCard
                param.socketType = Socket_Output;
            param.bInput = false;
            param.sockProp = Socket_Normal;
            param.wildCardGroup = param_desc.wildCard;
            ui.outputObjs.emplace_back(std::move(param));
        }
    }
    ParamTab tab;
    tab.groups.emplace_back(std::move(default));
    ui.inputPrims.emplace_back(std::move(tab));
    return ui;
}

ZENO_API void Session::defNodeClass(std::shared_ptr<INode>(*ctor)(), std::string const &clsname, Descriptor const &desc) {
    if (clsname == "CreateCube") {
        int j;
        j = 0;
    }
    
    if (nodeClasses.find(clsname) != nodeClasses.end()) {
        log_warn("node class redefined: `{}`\n", clsname);
        return;
    }

    CustomUI ui = descToCustomui(desc);
    auto cls = std::make_unique<ImplNodeClass>(ctor, ui, clsname);
    if (!clsname.empty() && clsname.front() == '^')
        return;

    std::string cate = cls->m_customui.category;
    if (m_cates.find(cate) == m_cates.end())
        m_cates.insert(std::make_pair(cate, std::vector<std::string>()));
    m_cates[cate].push_back(clsname);

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

ZENO_API void Session::defNodeReflectClass(std::function<std::shared_ptr<INode>()> ctor, zeno::reflect::TypeBase* pTypeBase)
{
    assert(pTypeBase);
    const zeno::reflect::ReflectedTypeInfo& info = pTypeBase->get_info();
    auto& nodecls = std::string(info.qualified_name.c_str());
    //有些name反射出来可能带有命名空间比如zeno::XXX
    int idx = nodecls.find_last_of(':');
    if (idx != std::string::npos) {
        nodecls = nodecls.substr(idx + 1);
    }

    if (nodeClasses.find(nodecls) != nodeClasses.end()) {
        //log_error("node class redefined: `{}`\n", nodecls);
        return;
    }
    auto cls = std::make_unique<ReflectNodeClass>(ctor, nodecls, pTypeBase);
    std::string cate = cls->m_customui.category;
    if (m_cates.find(cate) == m_cates.end())
        m_cates.insert(std::make_pair(cate, std::vector<std::string>()));
    m_cates[cate].push_back(nodecls);

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
    globalVariableManager.reset();
    globalVariableManager = std::make_unique<GlobalVariableManager>();
}

ZENO_API void Session::setApiLevelEnable(bool bEnable)
{
    m_bApiLevelEnable = bEnable;
}

ZENO_API void Session::beginApiCall()
{
    if (!m_bApiLevelEnable || m_bDisableRunning) return;
    m_apiLevel++;
}

ZENO_API void Session::endApiCall()
{
    if (!m_bApiLevelEnable || m_bDisableRunning) return;
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

ZENO_API void Session::setDisableRunning(bool bOn)
{
    m_bDisableRunning = bOn;
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

ZENO_API void Session::updateFrameRange(int start, int end)
{
    CORE_API_BATCH
    globalState->updateFrameRange(start, end);
}

ZENO_API void Session::interrupt() {
    m_bInterrupted = true;
}

ZENO_API bool Session::is_interrupted() const {
    return m_bInterrupted;
}

ZENO_API bool Session::run() {
    if (m_bDisableRunning)
        return false;

    if (m_bReentrance) {
        return true;
    }

    m_bReentrance = true;
    m_bInterrupted = false;
    globalState->set_working(true);

    objsMan->beforeRun();
    zeno::scope_exit sp([&]() { 
        objsMan->afterRun();
        m_bReentrance = false;
    });

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

static bool isBasedINode(const size_t hash) {
    static size_t inodecode = zeno::reflect::type_info<class zeno::INode>().hash_code();
    if (hash == inodecode)
        return true;

    auto& registry = zeno::reflect::ReflectionRegistry::get();
    auto typeHdl = registry->get(hash);
    const ArrayList<TypeHandle>& baseclasses = typeHdl->get_base_classes();
    for (auto& _typeHdl : baseclasses) {
        if (isBasedINode(_typeHdl.type_hash()))
            return true;
    }
    return false;
}

ZENO_API void Session::registerObjUIInfo(size_t hashcode, std::string_view color, std::string_view nametip) {
    s_objsUIInfo.insert(std::make_pair(hashcode, _ObjUIInfo { nametip, color }));
}

ZENO_API bool Session::getObjUIInfo(size_t hashcode, std::string_view& color, std::string_view& nametip) {
    auto iter = s_objsUIInfo.find(hashcode);
    if (iter == s_objsUIInfo.end()) {
        color = "#000000";
        nametip = "unknown type";
        return false;
    }
    color = iter->second.color;
    nametip = iter->second.name;
    return true;
}

ZENO_API void Session::initEnv(const zenoio::ZSG_PARSE_RESULT ioresult) {
    resetMainGraph();
    mainGraph->init(ioresult.mainGraph);
    //referManager->init(mainGraph);

    bool bDisableRun = m_bDisableRunning;
    m_bDisableRunning = true;
    scope_exit sp([&]() {m_bDisableRunning = bDisableRun; });
    switchToFrame(ioresult.timeline.currFrame);
    //init $F globalVariable
    //zeno::getSession().globalVariableManager->overrideVariable(zeno::GVariable("$F", zeno::reflect::make_any<float>(ioresult.timeline.currFrame)));
}

ZENO_API zeno::NodeCates Session::dumpCoreCates() {
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
        inputArray.PushBack(Value().SetString(zeno::paramTypeToString(input.type).c_str(), doc.GetAllocator()), doc.GetAllocator());
        inputArray.PushBack(Value().SetString(input.name.c_str(), doc.GetAllocator()), doc.GetAllocator());
        inputArray.PushBack(Value().SetString(input.defl.c_str(), doc.GetAllocator()), doc.GetAllocator());
        inputArray.PushBack(Value().SetString(input.doc.c_str(), doc.GetAllocator()), doc.GetAllocator());
        inputs.PushBack(inputArray, doc.GetAllocator());
    }

    // Outputs array
    Value outputs(kArrayType);
    for (const auto& output : descriptor.outputs) {
        Value outputArray(kArrayType);
        outputArray.PushBack(Value().SetString(zeno::paramTypeToString(output.type).c_str(), doc.GetAllocator()), doc.GetAllocator());
        outputArray.PushBack(Value().SetString(output.name.c_str(), doc.GetAllocator()), doc.GetAllocator());
        outputArray.PushBack(Value().SetString(output.defl.c_str(), doc.GetAllocator()), doc.GetAllocator());
        outputArray.PushBack(Value().SetString(output.doc.c_str(), doc.GetAllocator()), doc.GetAllocator());
        outputs.PushBack(outputArray, doc.GetAllocator());
    }

    // Params array
    Value params(kArrayType);
    for (const auto& param : descriptor.params) {
        Value paramArray(kArrayType);
        paramArray.PushBack(Value().SetString(zeno::paramTypeToString(param.type).c_str(), doc.GetAllocator()), doc.GetAllocator());
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
