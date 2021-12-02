#pragma once

#include "renderable.h"
#include <zeno/zty/mesh/Mesh.h>

ZENO_NAMESPACE_BEGIN

std::unique_ptr<Renderable> makeRenderableMesh(std::shared_ptr<zty::Mesh> const &mesh);

ZENO_NAMESPACE_END
