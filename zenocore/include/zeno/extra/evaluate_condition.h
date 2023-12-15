#pragma once

#include <zeno/types/ConditionObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/Exception.h>
#include <zeno/utils/vec.h>


namespace zeno {

static bool evaluate_condition(zeno::IObject *cond) {
    if (auto num = dynamic_cast<zeno::NumericObject *>(cond); num) {
        return std::visit([] (auto const &v) {
            return zeno::anytrue(v);
        }, num->value);
    } else if (auto con = dynamic_cast<zeno::ConditionObject *>(cond); con) {
        return con->get();
    } else {
        throw zeno::Exception("invalid input `"
                + (std::string)typeid(*cond).name() +
                "` to be evaluated as boolean");
    }
}

}
