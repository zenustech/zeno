#pragma once

#include <zeno/core/IObject.h>
#include <zeno/utils/api.h>
#include <zeno/core/Graph.h>
#include <zeno/extra/CAPI.h>

namespace zeno {

ZENO_API Zeno_Object capiLoadObjectSharedPtr(std::shared_ptr<IObject> const &objPtr_);
ZENO_API void capiEraseObjectSharedPtr(Zeno_Object object_);
ZENO_API std::shared_ptr<IObject> capiFindObjectSharedPtr(Zeno_Object object_);
ZENO_API Zeno_Graph capiLoadGraphSharedPtr(std::shared_ptr<Graph> const &graPtr_);
ZENO_API void capiEraseGraphSharedPtr(Zeno_Graph graph_);
ZENO_API std::shared_ptr<Graph> capiFindGraphSharedPtr(Zeno_Graph graph_);

}
