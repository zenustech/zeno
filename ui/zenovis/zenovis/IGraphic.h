#pragma once

#include <memory>
#include <zeno/core/IObject.h>

namespace zenovis {

struct Light;

struct IGraphic {
    virtual void draw(bool reflect, float depthPass) = 0;
    virtual void drawShadow(Light *light) = 0;
    virtual ~IGraphic() = default;
};

std::unique_ptr<IGraphic> makeGraphicPrimitive(std::shared_ptr<zeno::IObject> obj);

} // namespace zenovis
