#pragma once

#include <zeno/core/INode.h>
#include <zeno/core/IObject.h>
#include <zeno/core/Session.h>
#include <zeno/core/defNode.h>

//在这里引入反射生成的类型信息，注意zeno.h不要被INode.h Graph.h等核心头文件引用
#include "zeno_types/reflect/reflection.generated.hpp"
