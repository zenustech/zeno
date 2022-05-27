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
        int amount{0};
        std::vector<glm::mat4> modelMatrices;

        std::vector<float> timeList;
        float deltaTime{0.0f};
        std::vector<std::vector<zeno::vec3f>> vertexFrameBuffer;

        std::size_t serializeSize()
        {
            std::size_t size{0};

            size += sizeof(amount);

            auto modelMatricesSize{modelMatrices.size()};
            size += sizeof(modelMatricesSize);

            size += sizeof(modelMatrices[0]) * modelMatricesSize;

            auto timeListSize{timeList.size()};
            size += sizeof(timeListSize);

            size += sizeof(timeList[0]) * timeListSize;

            size += sizeof(deltaTime);

            auto frameAmount{vertexFrameBuffer.size()};
            size += sizeof(frameAmount);

            auto vertexAmount{vertexFrameBuffer[0].size()};
            size += sizeof(vertexAmount);

            size += sizeof(zeno::vec3f) * vertexAmount * frameAmount;

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

            auto timeListSize{timeList.size()};
            memcpy(str.data() + i, &timeListSize, sizeof(timeListSize));
            i += sizeof(timeListSize);

            memcpy(str.data() + i, timeList.data(), sizeof(timeList[0]) * timeListSize);
            i += sizeof(timeList[0]) * timeListSize;

            memcpy(str.data() + i, &deltaTime, sizeof(deltaTime));
            i += sizeof(deltaTime);

            auto frameAmount{vertexFrameBuffer.size()};
            memcpy(str.data() + i, &frameAmount, sizeof(frameAmount));
            i += sizeof(frameAmount);

            auto vertexAmount{vertexFrameBuffer[0].size()};
            memcpy(str.data() + i, &vertexAmount, sizeof(vertexAmount));
            i += sizeof(vertexAmount);

            for (int j = 0; j < frameAmount; ++j)
            {
                memcpy(str.data() + i, vertexFrameBuffer[j].data(), sizeof(zeno::vec3f) * vertexAmount);
                i += sizeof(zeno::vec3f) * vertexAmount;
            }

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

            size_t timeListSize;
            memcpy(&timeListSize, str.data() + i, sizeof(timeListSize));
            i += sizeof(timeListSize);
            inst.timeList.reserve(timeListSize);

            std::copy_n((float *)(str.data() + i), timeListSize, std::back_inserter(inst.timeList));
            i += sizeof(inst.timeList[0]) * timeListSize;

            memcpy(&inst.deltaTime, str.data() + i, sizeof(inst.deltaTime));
            i += sizeof(inst.deltaTime);

            size_t frameAmount;
            memcpy(&frameAmount, str.data() + i, sizeof(frameAmount));
            i += sizeof(frameAmount);

            size_t vertexAmount;
            memcpy(&vertexAmount, str.data() + i, sizeof(vertexAmount));
            i += sizeof(vertexAmount);

            auto &vertexFrameBuffer = inst.vertexFrameBuffer;
            vertexFrameBuffer.resize(frameAmount);
            for (int j = 0; j < frameAmount; ++j)
            {
                std::copy_n((zeno::vec3f *)(str.data() + i), vertexAmount, std::back_inserter(vertexFrameBuffer[j]));
                i += sizeof(zeno::vec3f) * vertexAmount;
            }

            return inst;
        }

    }; // struct InstancingObject

} // namespace zeno
