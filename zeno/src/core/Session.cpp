#include <zeno/core/Session.h>
#include <zeno/core/IObject.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/extra/GlobalComm.h>
#include <zeno/extra/GlobalStatus.h>
#include <zeno/extra/EventCallbacks.h>
#include <zeno/types/UserData.h>
#include <zeno/core/Graph.h>
#include <zeno/core/INode.h>
#include <zeno/utils/safe_at.h>
#include <zeno/utils/logger.h>
#include <zeno/utils/string.h>
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"

namespace zeno {

namespace {

struct ImplNodeClass : INodeClass {
    std::unique_ptr<INode>(*ctor)();

    ImplNodeClass(std::unique_ptr<INode>(*ctor)(), Descriptor const &desc)
        : INodeClass(desc), ctor(ctor) {}

    virtual std::unique_ptr<INode> new_instance() const override {
        return ctor();
    }
};

}

ZENO_API Session::Session()
    : globalState(std::make_unique<GlobalState>())
    , globalComm(std::make_unique<GlobalComm>())
    , globalStatus(std::make_unique<GlobalStatus>())
    , eventCallbacks(std::make_unique<EventCallbacks>())
    , m_userData(std::make_unique<UserData>())
    {
}

ZENO_API Session::~Session() = default;

ZENO_API void Session::defNodeClass(std::unique_ptr<INode>(*ctor)(), std::string const &id, Descriptor const &desc) {
    if (nodeClasses.find(id) != nodeClasses.end()) {
        log_error("node class redefined: `{}`\n", id);
    }
    auto cls = std::make_unique<ImplNodeClass>(ctor, desc);
    nodeClasses.emplace(id, std::move(cls));
}

//ZENO_API void Session::defOverloadNodeClass(
        //std::string const &id,
        //std::vector<std::string> const &types,
        //std::unique_ptr<INodeClass> &&cls) {
    //std::string key = '^' + id;
    //for (auto const &type: types) {
        //key += '^';
        //key += type;
    //}
    //defNodeClass(key, std::move(cls));
//}
//ZENO_API std::map<std::string, zany> callTempNode(std::string const &id,
        //std::map<std::string, zany> const &inputs) const {
    //auto se = getThisSession()->nodeClasses.at(id)->new_instance();
    //se->graph = getThisGraph();
    //se->inputs = inputs;
    //se->doApply();
    //return se->outputs;
//}

//ZENO_API std::unique_ptr<INode> Session::getOverloadNode(
        //std::string const &name, std::vector<std::shared_ptr<IObject>> const &inputs) {
    //std::string key = '^' + name;
    //for (auto const &obj: inputs) {
        //auto type = typeid(*obj).name();
        //key += '^';
        //key += type;
    //}
    //auto it = nodeClasses.find(key);
    //if (it == nodeClasses.end()) {
        //return nullptr;
    //}
    //auto const &cls = it->second;
    //auto node = cls->new_instance();
    //node->myname = key + "(TEMP)";

    //for (int i = 0; i < inputs.size(); i++) {
        //auto key = cls->desc->inputs.at(i).name;
        //node->inputs[key] = std::move(inputs[i]);
    //}
    //return node;
//}

ZENO_API INodeClass::INodeClass(Descriptor const &desc)
        : desc(std::make_unique<Descriptor>(desc)) {
}

ZENO_API INodeClass::~INodeClass() = default;

ZENO_API std::shared_ptr<Graph> Session::createGraph() {
    auto graph = std::make_shared<Graph>();
    graph->session = const_cast<Session *>(this);
    return graph;
}

ZENO_API std::string Session::dumpDescriptors() const {
    std::string res = "";
    std::vector<std::string> strs;

    for (auto const &[key, cls] : nodeClasses) {
        if (!key.empty() && key.front() == '^') continue; //overload nodes...
        res += "DESC@" + (key) + "@";
        Descriptor &desc = *cls->desc;

        strs.clear();
        for (auto const &[type, name, defl, _] : desc.inputs) {
            strs.push_back(type + "@" + (name) + "@" + defl);
        }
        res += "{" + join_str(strs, "%") + "}";
        strs.clear();
        for (auto const &[type, name, defl, _] : desc.outputs) {
            strs.push_back(type + "@" + (name) + "@" + defl);
        }
        res += "{" + join_str(strs, "%") + "}";
        strs.clear();
        for (auto const &[type, name, defl, _] : desc.params) {
            strs.push_back(type + "@" + (name) + "@" + defl);
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
