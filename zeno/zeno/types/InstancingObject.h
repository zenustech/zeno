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
        int frameAmount{0};
        int vertexAmount{0};
        std::vector<float> vertexFrameBuffer;

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

            size += sizeof(frameAmount);

            size += sizeof(vertexAmount);

            auto vertexFrameBufferSize{vertexFrameBuffer.size()};
            size += sizeof(vertexFrameBufferSize);

            size += sizeof(vertexFrameBuffer[0]) * vertexFrameBufferSize;

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

            memcpy(str.data() + i, &frameAmount, sizeof(frameAmount));
            i += sizeof(frameAmount);

            memcpy(str.data() + i, &vertexAmount, sizeof(vertexAmount));
            i += sizeof(vertexAmount);

            auto vertexFrameBufferSize{vertexFrameBuffer.size()};
            memcpy(str.data() + i, &vertexFrameBufferSize, sizeof(vertexFrameBufferSize));
            i += sizeof(vertexFrameBufferSize);

            memcpy(str.data() + i, vertexFrameBuffer.data(), sizeof(vertexFrameBuffer[0]) * vertexFrameBufferSize);
            i += sizeof(vertexFrameBuffer[0]) * vertexFrameBufferSize;

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

            memcpy(&inst.frameAmount, str.data() + i, sizeof(inst.frameAmount));
            i += sizeof(inst.frameAmount);

            memcpy(&inst.vertexAmount, str.data() + i, sizeof(inst.vertexAmount));
            i += sizeof(inst.vertexAmount);

            size_t vertexFrameBufferSize;
            memcpy(&vertexFrameBufferSize, str.data() + i, sizeof(vertexFrameBufferSize));
            i += sizeof(vertexFrameBufferSize);
            inst.vertexFrameBuffer.reserve(vertexFrameBufferSize);

            std::copy_n((float *)(str.data() + i), vertexFrameBufferSize, std::back_inserter(inst.vertexFrameBuffer));
            i += sizeof(inst.vertexFrameBuffer[0]) * vertexFrameBufferSize;

            return inst;
        }

    }; // struct InstancingObject

} // namespace zeno
