#pragma once

#include <zeno/core/IObject.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <iterator>

namespace zeno
{

    struct InstancingObject
        : IObjectClone<InstancingObject>
    {
        int amount;
        std::vector<glm::mat4> modelMatrices;

        std::size_t serializeSize()
        {
            std::size_t size{0};

            size += sizeof(amount);

            auto modelMatricesSize{modelMatrices.size()};
            size += sizeof(modelMatricesSize);

            size += sizeof(modelMatrices[0]) * modelMatricesSize;

            return size;
        }

        std::vector<char> serialize()
        {
            std::vector<char> str;
            str.resize(serializeSize());

            size_t i{0};

            memcpy(str.data() + i, &amount, sizeof(amount));
            i += sizeof(amount);

            auto modelMatricesSize{modelMatrices.size()};
            memcpy(str.data() + i, &modelMatricesSize, sizeof(modelMatricesSize));
            i += sizeof(modelMatricesSize);

            memcpy(str.data() + i, modelMatrices.data(), sizeof(modelMatrices[0]) * modelMatricesSize);
            i += sizeof(modelMatrices[0]) * modelMatricesSize;

            return str;
        }

        static InstancingObject deserialize(const std::vector<char> &str)
        {
            InstancingObject inst;

            size_t i{0};

            memcpy(&inst.amount, str.data() + i, sizeof(inst.amount));
            i += sizeof(inst.amount);

            size_t modelMatricesSize;
            memcpy(&modelMatricesSize, str.data() + i, sizeof(modelMatricesSize));
            i += sizeof(modelMatricesSize);
            inst.modelMatrices.reserve(modelMatricesSize);

            std::copy_n((glm::mat4 *)(str.data() + i), modelMatricesSize, std::back_inserter(inst.modelMatrices));
            i += sizeof(inst.modelMatrices[0]) * modelMatricesSize;

            return inst;
        }

    }; // struct InstancingObject

} // namespace zeno
