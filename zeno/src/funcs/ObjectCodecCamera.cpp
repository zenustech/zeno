#include <zeno/types/CameraObject.h>
#include <zeno/types/LightObject.h>
#include <zeno/funcs/ObjectCodec.h>
#include <zeno/utils/log.h>
#include <algorithm>
#include <cstring>

namespace zeno {

namespace _implObjectCodec {

std::shared_ptr<CameraObject> decodeCameraObject(const char *it);
std::shared_ptr<CameraObject> decodeCameraObject(const char *it) {
    auto obj = std::make_shared<CameraObject>();
    it = std::copy_n(it, sizeof(CameraData), (char *)static_cast<CameraData *>(obj.get()));
    return obj;
}

bool encodeCameraObject(CameraObject const *obj, std::back_insert_iterator<std::vector<char>> it);
bool encodeCameraObject(CameraObject const *obj, std::back_insert_iterator<std::vector<char>> it) {
    it = std::copy_n((char const *)static_cast<CameraData const *>(obj), sizeof(CameraData), it);
    return true;
}

std::shared_ptr<LightObject> decodeLightObject(const char *it);
std::shared_ptr<LightObject> decodeLightObject(const char *it) {
    auto obj = std::make_shared<LightObject>();
    it = std::copy_n(it, sizeof(LightData), (char *)static_cast<LightData *>(obj.get()));
    return obj;
}

bool encodeLightObject(LightObject const *obj, std::back_insert_iterator<std::vector<char>> it);
bool encodeLightObject(LightObject const *obj, std::back_insert_iterator<std::vector<char>> it) {
    it = std::copy_n((char const *)static_cast<LightData const *>(obj), sizeof(LightData), it);
    return true;
}

}

}
