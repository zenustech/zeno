#include <zeno/core/Assets.h>


namespace zeno {

ZENO_API Assets::Assets() {

}

ZENO_API Assets::~Assets() {

}

ZENO_API void Assets::createAsset(const std::string& name) {
    std::shared_ptr<Graph> newAsst = std::make_shared<Graph>(name);
    assets.insert(std::make_pair(name, newAsst));
    CALLBACK_NOTIFY(createAsset, name)
}

ZENO_API void Assets::removeAsset(const std::string& name) {
    assets.erase(name);
    CALLBACK_NOTIFY(removeAsset, name)
}

ZENO_API void Assets::renameAsset(const std::string& old_name, const std::string& new_name) {
    //TODO
    CALLBACK_NOTIFY(renameAsset, old_name, new_name)
}

ZENO_API std::shared_ptr<Graph> Assets::getAsset(const std::string& name) const {
    return assets.at(name);
}

ZENO_API NodeData Assets::newInstance(const std::string& name) {
    NodeData data;
    return data;
}

}