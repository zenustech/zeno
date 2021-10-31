#pragma once


#include <zeno/dop/Functor.h>
#include <zeno/dop/Node.h>


ZENO_NAMESPACE_BEGIN
namespace dop {


struct OverloadNode : Node {
    virtual void apply() override;
};


}
ZENO_NAMESPACE_END
