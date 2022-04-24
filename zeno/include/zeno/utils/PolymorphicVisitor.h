#pragma once

#include <utility>
#include <type_traits>

namespace zeno {

#define _ZENO_DEFINE_VISITOR_HELPER(Derived) \
        virtual void visit(Derived *) = 0;

#define ZENO_DEFINE_VISITOR(VisitorName, Base, BaseXMacro) \
    struct VisitorName { \
        BaseXMacro(_ZENO_DEFINE_VISITOR_HELPER, ); \
    };

}
