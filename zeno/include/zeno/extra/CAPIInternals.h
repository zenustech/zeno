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
ZENO_API int capiRegisterObjectFactory(std::string const &typeName_, Zeno_Object (*factory_)(void *));
ZENO_API int capiRegisterObjectDefactory(std::string const &typeName_, void *(*defactory_)(Zeno_Object));
ZENO_API int capiRegisterCFunctionPtr(std::string const &typeName_, void *(*cfunc_)(void *));
ZENO_API Zeno_Error capiLastErrorCatched(std::function<void()> const &func) noexcept;

}
