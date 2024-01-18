#ifndef __CORE_ASSETS_H__
#define __CORE_ASSETS_H__

#include <zeno/core/Graph.h>

namespace zeno {

struct Assets : std::enable_shared_from_this<Assets> {
    Session *session = nullptr;

    std::map<std::string, std::shared_ptr<Graph>> assets;

    ZENO_API Assets();
    ZENO_API ~Assets();

    Assets(Assets const&) = delete;
    Assets& operator=(Assets const&) = delete;
    Assets(Assets&&) = delete;
    Assets& operator=(Assets&&) = delete;

    ZENO_API void createAsset(const std::string& name);
    CALLBACK_REGIST(createAsset, void, const std::string&)

    ZENO_API void removeAsset(const std::string& name);
    CALLBACK_REGIST(removeAsset, void, const std::string&)

    ZENO_API void renameAsset(const std::string& old_name, const std::string& new_name);
    CALLBACK_REGIST(renameAsset, void, const std::string&, const std::string&)

    ZENO_API std::shared_ptr<Graph> getAsset(const std::string& name) const;
    ZENO_API NodeData newInstance(const std::string& name);
};

}

#endif