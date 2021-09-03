#include <zeno/core/Session.h>
#include <zeno/core/IObject.h>
#include <zeno/core/Scene.h>
#include <zeno/core/INode.h>
#include <zeno/utils/safe_at.h>
#include <spdlog/spdlog.h>
#include <sstream>

namespace zeno {

ZENO_API Session::Session() = default;
ZENO_API Session::~Session() = default;

ZENO_API void Session::defNodeClass(std::string const &id, std::unique_ptr<INodeClass> &&cls) {
    if (nodeClasses.find(id) != nodeClasses.end()) {
        spdlog::warn("node class redefined: `{}`\n", id);
    }
    nodeClasses.emplace(id, std::move(cls));
}

ZENO_API void Session::defOverloadNodeClass(
        std::string const &id,
        std::vector<std::string> const &types,
        std::unique_ptr<INodeClass> &&cls) {
    std::string key = '@' + id;
    for (auto const &type: types) {
        key += '@';
        key += type;
    }
    defNodeClass(key, std::move(cls));
}

ZENO_API std::unique_ptr<INode> Session::getOverloadNode(
        std::string const &name, std::vector<std::shared_ptr<IObject>> const &args) {
    std::string key = '@' + name;
    for (auto const &obj: args) {
        auto type = typeid(*obj).name();
        key += '@';
        key += type;
    }
    auto cls = safe_at(nodeClasses, key, "object method");
    auto node = cls->new_instance();

    for (int i = 0; i < args.size(); i++) {
        std::stringstream ss;
        ss << "overload_" << i;
        node->inputs[ss.str()] = std::move(args[i]);
    }
    return node;
}

ZENO_API INodeClass::INodeClass(Descriptor const &desc)
        : desc(std::make_unique<Descriptor>(desc)) {
}

ZENO_API INodeClass::~INodeClass() = default;

ZENO_API std::unique_ptr<Scene> Session::createScene() {
    auto scene = std::make_unique<Scene>();
    scene->sess = const_cast<Session *>(this);
    return scene;
}

ZENO_API Scene &Session::getDefaultScene() {
    if (!defaultScene)
        defaultScene = createScene();
    return *defaultScene;
}

ZENO_API std::string Session::dumpDescriptors() const {
  std::string res = "";
  for (auto const &[key, cls] : nodeClasses) {
    res += "DESC@" + key + "@" + cls->desc->serialize() + "\n";
  }
  return res;
}


ZENO_API Session &getSession() {
    static std::unique_ptr<Session> ptr;
    if (!ptr) {
        ptr = std::make_unique<Session>();
    }
    return *ptr;
}

}
