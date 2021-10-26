#pragma once

#include "renderable.h"

ZENO_NAMESPACE_BEGIN

std::unique_ptr<Renderable> makeRenderableMesh(std::shared_ptr<zeno::types::Mesh> const &mesh)

ZENO_NAMESPACE_END
