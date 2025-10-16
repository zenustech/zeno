#include <zeno/types/NumericObject.h>

namespace zeno {

    NumericObject::NumericObject(NumericValue const& value) : value(value) {}

    void NumericObject::Delete() {
        IObjectClone<NumericObject>::Delete();
    }

    NumericObject::~NumericObject() {

    }

}