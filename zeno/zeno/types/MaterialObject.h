#pragma once

#include <zeno/core/IObject.h>
#include <string>
#include <cstring>
#include <cassert>

namespace zeno
{

    struct MaterialObject
        : IObjectClone<MaterialObject>
    {
        std::string vert;
        std::string frag;

        size_t serialize(char buff[])
        {
            size_t i{0};

            auto vertLen{vert.size()};
            memcpy(buff + i, &vertLen, sizeof(vertLen));
            i += sizeof(vertLen);

            vert.copy(buff + i, vertLen);
            i += vertLen;

            auto fragLen{frag.size()};
            memcpy(buff + i, &fragLen, sizeof(fragLen));
            i += sizeof(fragLen);

            frag.copy(buff + i, fragLen);
            i += fragLen;

            return i + 1;
        }

        static MaterialObject deserialize(char buff[], size_t size)
        {
            MaterialObject mtl;

            size_t i{0};

            size_t vertLen;
            memcpy(&vertLen, buff + i, sizeof(vertLen));
            i += sizeof(vertLen);

            mtl.vert = {std::string{buff + i, vertLen}};
            i += vertLen;

            size_t fragLen;
            memcpy(&fragLen, buff + i, sizeof(fragLen));
            i += sizeof(fragLen);

            mtl.frag = {std::string{buff + i, fragLen}};
            i += fragLen;

            return mtl;
        }

    }; // struct Material

} // namespace zeno
