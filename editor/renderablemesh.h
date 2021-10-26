#pragma once

#include "renderable.h"

std::unique_ptr<Renderable> makeRenderableMesh(std::shared_ptr<zeno::types::Mesh> const &mesh)
