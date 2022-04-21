#include <zeno/types/CameraObject.h>
#include <zeno/funcs/ObjectCodec.h>
#include <zeno/utils/log.h>
#include <algorithm>
#include <cstring>

namespace zeno {

namespace _implObjectCodec {

std::shared_ptr<CameraObject> decodeCameraObject(const char *it);
std::shared_ptr<CameraObject> decodeCameraObject(const char *it) {
    auto obj = std::make_shared<CameraObject>();
    it = std::copy_n(it, sizeof(*obj), (char *)obj.get());
    return obj;
}

bool encodeCameraObject(CameraObject const *obj, std::back_insert_iterator<std::vector<char>> it);
bool encodeCameraObject(CameraObject const *obj, std::back_insert_iterator<std::vector<char>> it) {
    it = std::copy_n((char const *)obj, sizeof(*obj), it);
    return true;
}

}

}
