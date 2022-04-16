#pragma once

#include <memory>

namespace zenovis {

struct Light;

struct IGraphic {
    virtual void draw(bool reflect, float depthPass) = 0;
    virtual void drawShadow(Light *light) = 0;
    virtual ~IGraphic() = default;
};

} // namespace zenovis
