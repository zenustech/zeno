#pragma once

#include <zeno/types/PrimitiveObject.h>
#include <tinygltf/json.hpp>
#include <glm/glm.hpp>
#include <array>

namespace zeno {

// Prepared representations retain their ordinary primitive payload for other
// consumers and for cache serialization. The additional storage is immutable.
struct PreparedMatrixStorage {
    std::vector<std::array<float, 12>> transforms;
    std::vector<glm::mat4> uiTransforms;
    std::vector<int> instanceIds;
};

struct PreparedMatrix : PrimitiveObject {
    std::shared_ptr<const PreparedMatrixStorage> storage;
    size_t offset=0, count=0, idOffset=0, idCount=0;
    std::string matrixName;
    bool unchanged=false;
};

struct PreparedMatrixBatch : IObjectClone<PreparedMatrixBatch> {
    std::vector<std::pair<std::string,std::shared_ptr<PreparedMatrix>>> matrices;
};

struct PreparedSceneTree : PrimitiveObject {
    std::shared_ptr<const nlohmann::json> topology;
};

}
