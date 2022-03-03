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
        std::string common;
        std::string extensions;

        std::vector<char> serialize()
        {
            std::vector<char> str;

            size_t i{0};

            auto vertLen{vert.size()};
            auto fragLen{frag.size()};
            auto commonLen{common.size()};
            auto extensionsLen{extensions.size()};
            str.resize(vertLen + sizeof(vertLen) + fragLen + sizeof(fragLen)
                       + commonLen + sizeof(commonLen) + extensionsLen + sizeof(extensionsLen));

            memcpy(str.data() + i, &vertLen, sizeof(vertLen));
            i += sizeof(vertLen);

            vert.copy(str.data() + i, vertLen);
            i += vertLen;

            memcpy(str.data() + i, &fragLen, sizeof(fragLen));
            i += sizeof(fragLen);

            frag.copy(str.data() + i, fragLen);
            i += fragLen;

            memcpy(str.data() + i, &commonLen, sizeof(commonLen));
            i += sizeof(commonLen);

            common.copy(str.data() + i, commonLen);
            i += commonLen;

            memcpy(str.data() + i, &extensionsLen, sizeof(extensionsLen));
            i += sizeof(extensionsLen);

            extensions.copy(str.data() + i, extensionsLen);
            i += extensionsLen;

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

            size_t commonLen;
            memcpy(&commonLen, str.data() + i, sizeof(commonLen));
            i += sizeof(commonLen);

            mtl.common = std::string{str.data() + i, commonLen};
            i += commonLen;

            size_t extensionsLen;
            memcpy(&extensionsLen, str.data() + i, sizeof(extensionsLen));
            i += sizeof(extensionsLen);

            mtl.extensions = std::string{str.data() + i, extensionsLen};
            i += extensionsLen;

            return mtl;
        }

    }; // struct Material

} // namespace zeno
