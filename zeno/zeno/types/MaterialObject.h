#pragma once

#include <zeno/core/IObject.h>
#include <string>
#include <cstring>
#include <vector>

namespace zeno
{

    struct MaterialObject
        : IObjectClone<MaterialObject>
    {
        std::string vert;
        std::string frag;

        std::vector<char> serialize()
        {
            std::vector<char> str;

            size_t i{0};

            auto vertLen{vert.size()};
            auto fragLen{frag.size()};
            str.resize(vertLen + sizeof(vertLen) + fragLen + sizeof(fragLen));

            memcpy(str.data() + i, &vertLen, sizeof(vertLen));
            i += sizeof(vertLen);

            vert.copy(str.data() + i, vertLen);
            i += vertLen;

            memcpy(str.data() + i, &fragLen, sizeof(fragLen));
            i += sizeof(fragLen);

            frag.copy(str.data() + i, fragLen);
            i += fragLen;

            return str;
        }

        static MaterialObject deserialize(std::vector<char> str)
        {
            MaterialObject mtl;

            size_t i{0};

            size_t vertLen;
            memcpy(&vertLen, str.data() + i, sizeof(vertLen));
            i += sizeof(vertLen);

            mtl.vert = std::string{str.data() + i, vertLen};
            i += vertLen;

            size_t fragLen;
            memcpy(&fragLen, str.data() + i, sizeof(fragLen));
            i += sizeof(fragLen);

            mtl.frag = std::string{str.data() + i, fragLen};
            i += fragLen;

            return mtl;
        }

    }; // struct Material

} // namespace zeno
