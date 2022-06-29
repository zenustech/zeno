#ifndef ZENO_READOBJPRIM_H
#define ZENO_READOBJPRIM_H
#include <zeno/utils/vec.h>
#include <zeno/types/PrimitiveObject.h>

std::shared_ptr<zeno::PrimitiveObject> wenqiang_parse_obj(std::vector<char> &&bin);


#endif //ZENO_READOBJPRIM_H
