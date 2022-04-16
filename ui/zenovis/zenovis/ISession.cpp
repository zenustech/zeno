#include <zenovis/ISession.h>
#include <zenovis/IGraphic.h>
#include <zenovis/Scene.h>
#include <zeno/core/IObject.h>

namespace zenovis {

extern std::unique_ptr<IGraphic> makeGraphic(std::shared_ptr<zeno::IObject> obj);

struct ImplSession : Isession {
    std::unique_ptr<Scene> scene = std::make_unique<Scene>();

    ImplSession() = default;
    ~ImplSession() = default;
};

std::unique_ptr<ISession> makeSession() {
    return std::make_unique<ImplSession>();
}

}
