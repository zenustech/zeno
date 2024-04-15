#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveUtils.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/orthonormal.h>
#include <zeno/utils/variantswitch.h>
#include <zeno/utils/ticktock.h>
#include <zeno/utils/parallel_reduce.h>
#include <sstream>
#include <iostream>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace zeno {

struct ElementCollapse : zeno::INode {
    virtual void apply() override {

    }
};

}
