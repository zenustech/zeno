#pragma once

#include <zenovis/Camera.h>
#include <zenovis/Scene.h>
#include <type_traits>
#include <functional>
#include <string>
#include <memory>
#include <map>
#include "tinygltf/json.hpp"
using Json = nlohmann::json;

namespace zenovis {

struct Scene;

struct RenderEngine {
    virtual void draw(bool record) = 0;
    virtual void update() = 0;
    virtual void assetLoad() = 0;
    virtual void run() = 0;
    virtual void beginFrameLoading(int frameid) = 0;
    virtual void endFrameLoading(int frameid) = 0;
    virtual void cleanupAssets() = 0;
    virtual void cleanupWhenExit() = 0;

    virtual ~RenderEngine() = default;
    virtual std::optional<glm::vec3> getClickedPos(float x, float y) { return {}; }
    virtual std::optional<std::tuple<std::string, uint32_t, uint32_t>> getClickedId(float x, float y) { return {}; }
    virtual void load_matrix_objects(std::vector<std::shared_ptr<zeno::IObject>> matrixs) {};
    virtual void outlineInit(Json const &msg) {};

    virtual void showBackground(bool bShow) {};

    std::function<void(std::string)> fun = [](std::string){};
};

class RenderManager {
    static std::map<std::string, std::function<std::unique_ptr<RenderEngine>(Scene *)>> factories;
    std::map<std::string, std::unique_ptr<RenderEngine>> instances;
    std::string defaultEngineName;
    Scene *scene;

public:
    explicit RenderManager(Scene *scene_) : scene(scene_) {
    }

    template <class T, class = std::enable_if_t<std::is_base_of_v<RenderEngine, T>>>
    static int registerRenderEngine(std::string const &name) {
        factories.emplace(name, [] (Scene *s) { return std::make_unique<T>(s); });
        return 1;
    } 

    RenderEngine *getEngine(std::string const &name) {
        auto it = instances.find(name);
        if (it == instances.end()) {
            it = instances.emplace(name, factories.at(name)(scene)).first;
        }
        return it->second.get();
    }

    RenderEngine *getEngine() {
        return getEngine(defaultEngineName);
    }

    std::string getDefaultEngineName() const {
        return defaultEngineName;
    }

    void switchDefaultEngine(std::string const &name) {
        defaultEngineName = name;
    }
};

}
