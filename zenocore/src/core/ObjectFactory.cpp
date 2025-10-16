#include <zeno/types/PrimitiveObject.h>


namespace zeno {

ZENO_API PrimitiveObject* createPrimitive() {
    return new PrimitiveObject;
}

ZENO_API PrimitiveObject* clonePrimitive(PrimitiveObject* pObj) {
    return new PrimitiveObject(*pObj);
}

}