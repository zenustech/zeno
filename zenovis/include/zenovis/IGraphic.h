#pragma once

#include <memory>

namespace zenovis {

struct Light;

struct IGraphic {
    virtual void draw(bool reflect, bool depthPass) = 0;
    virtual void drawShadow(Light *light) = 0;
    virtual bool hasMaterial() const { return false; }
    virtual ~IGraphic() = default;
};

} // namespace zenovis
