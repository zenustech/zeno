#pragma once

#include <zeno/core/IObject.h>
#include <zeno/types/TextureObject.h>
#include <string>
#include <cstring>
#include <vector>
#include <iostream>

namespace zeno
{

    struct MaterialObject
        : IObjectClone<MaterialObject>
    {
        std::string vert;
        std::string frag;
        std::string common;
        std::string extensions;
        std::vector<std::shared_ptr<Texture2DObject>> tex2Ds;
        int mtlid;

        size_t serializeSize()
        {
            size_t size{0};

            size += sizeof(mtlid);

            auto vertLen{vert.size()};
            size += sizeof(vertLen);
            size += vertLen;

            auto fragLen{frag.size()};
            size += sizeof(fragLen);
            size += fragLen;

            auto commonLen{common.size()};
            size += sizeof(commonLen);
            size += commonLen;

            auto extensionsLen{extensions.size()};
            size += sizeof(fragLen);
            size += fragLen;

            auto tex2DsSize{tex2Ds.size()};
            size += sizeof(tex2DsSize);
            for (const auto &tex2D : tex2Ds)
            {
                auto tex2DStrSize = tex2D->serializeSize();
                size += sizeof(tex2DStrSize);
                size += tex2DStrSize;
            }

            return size;
        }

        void serialize(char *str)
        {
            size_t i{0};

            memcpy(str + i, &mtlid, sizeof(mtlid));
            i += mtlid;

            auto vertLen{vert.size()};
            memcpy(str + i, &vertLen, sizeof(vertLen));
            i += sizeof(vertLen);

            vert.copy(str + i, vertLen);
            i += vertLen;

            auto fragLen{frag.size()};
            memcpy(str + i, &fragLen, sizeof(fragLen));
            i += sizeof(fragLen);

            frag.copy(str + i, fragLen);
            i += fragLen;

            auto commonLen{common.size()};
            memcpy(str + i, &commonLen, sizeof(commonLen));
            i += sizeof(commonLen);

            common.copy(str + i, commonLen);
            i += commonLen;

            auto extensionsLen{extensions.size()};
            memcpy(str + i, &extensionsLen, sizeof(extensionsLen));
            i += sizeof(extensionsLen);

            extensions.copy(str + i, extensionsLen);
            i += extensionsLen;

            auto tex2DsSize{tex2Ds.size()};
            memcpy(str + i, &tex2DsSize, sizeof(tex2DsSize));
            i += sizeof(tex2DsSize);

            for (const auto &tex2D : tex2Ds)
            {
                auto tex2DStr = tex2D->serialize();
                auto tex2DStrSize = tex2DStr.size();
                memcpy(str + i, &tex2DStrSize, sizeof(tex2DStrSize));
                i += sizeof(tex2DStrSize);

                memcpy(str + i, tex2DStr.data(), tex2DStrSize);
                i += tex2DStrSize;
            }
        }

        std::vector<char> serialize()
        {
            std::vector<char> str(serializeSize());
            serialize(str.data());
            return str;
        }

        void deserialize(const char *str)
        {
            size_t i{0};

            memcpy(&mtlid, str + i, sizeof(mtlid));
            i += mtlid;

            size_t vertLen;
            memcpy(&vertLen, str + i, sizeof(vertLen));
            i += sizeof(vertLen);

            this->vert = std::string{str + i, vertLen};
            i += vertLen;

            size_t fragLen;
            memcpy(&fragLen, str + i, sizeof(fragLen));
            i += sizeof(fragLen);

            this->frag = std::string{str + i, fragLen};
            i += fragLen;

            size_t commonLen;
            memcpy(&commonLen, str + i, sizeof(commonLen));
            i += sizeof(commonLen);

            this->common = std::string{str + i, commonLen};
            i += commonLen;

            size_t extensionsLen;
            memcpy(&extensionsLen, str + i, sizeof(extensionsLen));
            i += sizeof(extensionsLen);

            this->extensions = std::string{str + i, extensionsLen};
            i += extensionsLen;

            size_t tex2DsSize;
            memcpy(&tex2DsSize, str + i, sizeof(tex2DsSize));
            i += sizeof(tex2DsSize);
            this->tex2Ds.resize(tex2DsSize);

            for (size_t j{0}; j < tex2DsSize; ++j)

            {
                size_t tex2DStrSize;
                memcpy(&tex2DStrSize, str + i, sizeof(tex2DStrSize));
                i += sizeof(tex2DStrSize);

                std::vector<char> tex2DStr;
                tex2DStr.resize(tex2DStrSize);
                memcpy(tex2DStr.data(), str + i, tex2DStrSize);
                i += tex2DStrSize;

                auto tex2D = std::make_shared<Texture2DObject>(
                    Texture2DObject::deserialize(tex2DStr));
                this->tex2Ds[j] = tex2D;
            }
        }

        static MaterialObject deserialize(const std::vector<char> &str)
        {
            MaterialObject mtl;
            mtl.deserialize(str.data());
            return mtl;
        }

    }; // struct Material

} // namespace zeno
