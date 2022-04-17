#pragma once

#include <zeno/core/IObject.h>
#include <zenovis/IGraphic.h>

namespace zenovis {

struct Scene;

std::unique_ptr<IGraphic> makeGraphic(Scene *scene, std::shared_ptr<zeno::IObject> obj);

}
