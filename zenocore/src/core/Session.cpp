#include <zeno/core/Session.h>
#include <zeno/core/IObject.h>
#include <zeno/core/Assets.h>
#include <zeno/core/ObjectManager.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/extra/GlobalComm.h>
#include <zeno/extra/GlobalStatus.h>
#include <zeno/extra/EventCallbacks.h>
#include <zeno/types/UserData.h>
#include <zeno/core/Graph.h>
#include <zeno/core/INode.h>
#include <zeno/core/IParam.h>
#include <zeno/core/CalcManager.h>
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

    ImplNodeClass(std::shared_ptr<INode>(*ctor)(), Descriptor const &desc, std::string const &name)
        : INodeClass(desc, name), ctor(ctor) {}

    virtual std::shared_ptr<INode> new_instance(Graph* pGraph, std::string const &name) const override {
        std::shared_ptr<INode> spNode = ctor();
        spNode->initUuid(pGraph, classname);

        std::vector<SocketDescriptor> inputs = desc->inputs;
        std::vector<ParamDescriptor> params = desc->params;
        std::vector<SocketDescriptor> outputs = desc->outputs;

        std::shared_ptr<SubnetNode> spSubnet = std::dynamic_pointer_cast<SubnetNode>(spNode);

        spNode->set_name(name);

        //init all params, and set defl value
        for (SocketDescriptor& param_desc : inputs)
        {
            std::shared_ptr<IParam> sparam = std::make_shared<IParam>();
            sparam->name = param_desc.name;
            sparam->m_wpNode = spNode;
            sparam->type = zeno::convertToType(param_desc.type);
            sparam->defl = zeno::str2var(param_desc.defl, sparam->type);
            sparam->socketType = param_desc.socketType;
            if (param_desc.control != NullControl)
                sparam->control = param_desc.control;
            if (starts_with(param_desc.type, "enum ")) {
                //compatible with old case of combobox items.
                sparam->type = Param_String;
                sparam->control = Combobox;
                std::vector<std::string> items = split_str(param_desc.type, ' ');
                if (!items.empty()) {
                    items.erase(items.begin());
                    ControlProperty props = ControlProperty();
                    props.items = items;
                    sparam->optCtrlprops = props;
                }
            }
            spNode->add_input_param(sparam);
        }

        for (ParamDescriptor& param_desc : params)
        {
            std::shared_ptr<IParam> sparam = std::make_shared<IParam>();
            sparam->name = param_desc.name;
            sparam->m_wpNode = spNode;
            sparam->type = zeno::convertToType(param_desc.type);

            if (starts_with(param_desc.type, "enum ")) {
                //compatible with old case of combobox items.
                sparam->type = Param_String;
                sparam->control = Combobox;
                std::vector<std::string> items = split_str(param_desc.type, ' ');
                if (!items.empty()) {
                    items.erase(items.begin());
                    ControlProperty props = ControlProperty();
                    props.items = items;
                    sparam->optCtrlprops = props;
                }
            }

            sparam->defl = zeno::str2var(param_desc.defl, sparam->type);
            sparam->socketType = NoSocket;
            spNode->add_input_param(sparam);
        }

        for (SocketDescriptor& param_desc : outputs)
        {
            std::shared_ptr<IParam> sparam = std::make_shared<IParam>();
            sparam->name = param_desc.name;
            sparam->m_wpNode = spNode;
            sparam->type = zeno::convertToType(param_desc.type);
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
    , globalStatus(std::make_unique<GlobalStatus>())
    , eventCallbacks(std::make_unique<EventCallbacks>())
    , m_userData(std::make_unique<UserData>())
    , mainGraph(std::make_shared<Graph>("main"))
    , assets(std::make_shared<AssetsMgr>())
    , objsMan(std::make_unique<ObjectManager>())
    , calcMan(std::make_unique<CalcManager>())
{
    initNodeCates();
}

ZENO_API Session::~Session() = default;

ZENO_API void Session::defNodeClass(std::shared_ptr<INode>(*ctor)(), std::string const &clsname, Descriptor const &desc) {
    if (nodeClasses.find(clsname) != nodeClasses.end()) {
        log_error("node class redefined: `{}`\n", clsname);
    }
    auto cls = std::make_unique<ImplNodeClass>(ctor, desc, clsname);
    nodeClasses.emplace(clsname, std::move(cls));
}

ZENO_API INodeClass::INodeClass(Descriptor const &desc, std::string const& classname)
        : desc(std::make_unique<Descriptor>(desc))
        , classname(classname){
}

ZENO_API INodeClass::~INodeClass() = default;

ZENO_API std::shared_ptr<Graph> Session::createGraph(const std::string& name) {
    auto graph = std::make_shared<Graph>(name);
    return graph;
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
        //TODO: always mode
        if (m_bAutoRun)
            run();
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
    calcMan->mark_frame_change_dirty();
    globalState->frameid = frameid;
}

ZENO_API bool Session::run() {
    calcMan->run();
    return true;
}

ZENO_API void Session::set_auto_run(bool bOn) {
    m_bAutoRun = bOn;
}

ZENO_API bool Session::is_auto_run() const {
    return m_bAutoRun;
}

void Session::initNodeCates() {
    for (auto const& [key, cls] : nodeClasses) {
        if (!key.empty() && key.front() == '^')
            continue;
        Descriptor& desc = *cls->desc;
        for (std::string cate : desc.categories) {
            if (m_cates.find(cate) == m_cates.end())
                m_cates.insert(std::make_pair(cate, std::vector<std::string>()));
            m_cates[cate].push_back(key);
        }
    }
}

ZENO_API zeno::NodeCates Session::dumpCoreCates() {
    if (m_cates.empty()) {
        initNodeCates();
    }
    return m_cates;
}

ZENO_API std::string Session::dumpDescriptors() const {
    std::string res = "";
    std::vector<std::string> strs;

    for (auto const &[key, cls] : nodeClasses) {
        if (!key.empty() && key.front() == '^') continue; //overload nodes...
        res += "DESC@" + (key) + "@";
        Descriptor &desc = *cls->desc;

        strs.clear();
        for (auto const &desc : desc.inputs) {
            strs.push_back(desc.type + "@" + (desc.name) + "@" + desc.defl);
        }
        res += "{" + join_str(strs, "%") + "}";
        strs.clear();
        for (auto const &desc : desc.outputs) {
            strs.push_back(desc.type + "@" + (desc.name) + "@" + desc.defl);
        }
        res += "{" + join_str(strs, "%") + "}";
        strs.clear();
        for (auto const &desc : desc.params) {
            strs.push_back(desc.type + "@" + (desc.name) + "@" + desc.defl);
        }
        res += "{" + join_str(strs, "%") + "}";
        res += "{" + join_str(desc.categories, "%") + "}";

        res += "\n";
    }
    return res;
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
    std::string res = "";
    std::vector<std::string> strs;

    for (auto const &[key, cls] : nodeClasses) {
        res += dumpDescriptorToJson(key, *cls->desc);
        res += "\n";
    }
    return res;
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
