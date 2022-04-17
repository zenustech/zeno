#include <unordered_map>
#include <vector>
#include <zeno/utils/log.h>
#include <zenovis/makeGraphic.h>
#include <zenovis/Scene.h>

namespace zenovis {

struct GraphicsManager {
    Scene *scene;

    std::unordered_map<std::shared_ptr<zeno::IObject>,
                       std::unique_ptr<IGraphic>>
        graphics, new_graphics;

    explicit GraphicsManager(Scene *scene) : scene(scene) {
    }

    void load_objects(std::vector<std::shared_ptr<zeno::IObject>> const &objs) {
        new_graphics.clear();
        for (auto const &obj : objs) {
            zeno::log_trace("load_object: got object {}", obj.get());
            auto it = graphics.find(obj);
            if (it != graphics.end()) {
                zeno::log_trace("load_object: cache hit graphics {}",
                                it->second.get());
                new_graphics.emplace(it->first, std::move(it->second));
                graphics.erase(it);
                continue;
            }
            auto ig = makeGraphic(scene, obj);
            zeno::log_trace("load_object: fresh load graphics {}", ig.get());
            if (!ig)
                continue;
            new_graphics.emplace(obj, std::move(ig));
        }
        std::swap(new_graphics, graphics);
        new_graphics.clear();
    }
};

} // namespace zenovis
