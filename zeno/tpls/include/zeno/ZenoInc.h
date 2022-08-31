#pragma once

#ifndef _MSC_VER
#warning "<zeno/ZenoInc.h> is deprecated, use multiple #include's directly instead"
#endif

#include <zeno/zeno.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/FunctionObject.h>
#include <zeno/types/DictObject.h>
#include <zeno/types/GenericObject.h>
#include <zeno/types/DummyObject.h>
#include <zeno/core/Graph.h>
#include <zeno/extra/TempNode.h>
#include <zeno/extra/MethodCaller.h>
#include <zeno/types/UserData.h>
#include <zeno/utils/log.h>
#include <zeno/utils/zeno_p.h>
