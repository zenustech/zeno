#pragma once

#include <zeno/core/IObject.h>
#include <zeno/utils/PolymorphicMap.h>
#include <zeno/utils/api.h>
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <map>
#include <set>
#include <functional>

namespace zeno {

struct GlobalComm {

    static void toDisk(std::string cachedir, int frameid, std::map<std::string, zany>& objs, bool cacheLightCameraOnly, bool cacheMaterialOnly, std::string fileName = "");
    static bool fromDisk(std::string cachedir, int frameid, std::map<std::string, zany>& objs, std::string fileName = "");
};

}
