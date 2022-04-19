#pragma once

#include <zeno/core/IObject.h>
#include <zenovis/IGraphic.h>

namespace zenovis {

struct Scene;

std::unique_ptr<IGraphic> makeGraphic(Scene *scene,
                                      std::shared_ptr<zeno::IObject> obj);
std::unique_ptr<IGraphic> makeGraphicAxis(Scene *scene);
std::unique_ptr<IGraphic> makeGraphicGrid(Scene *scene);

} // namespace zenovis
