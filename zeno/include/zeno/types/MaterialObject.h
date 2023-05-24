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
        std::vector<std::pair<std::string, std::string>> tex3Ds;
        std::string transform;
        std::string mtlidkey;  // unused for now

        size_t serializeSize() const
        {
            size_t size{0};

            auto mtlidkeyLen{mtlidkey.size()};
            size += sizeof(mtlidkeyLen);
            size += mtlidkeyLen;

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
            size += sizeof(extensionsLen);
            size += extensionsLen;

            auto tex2DsSize{tex2Ds.size()};
            size += sizeof(tex2DsSize);
            for (const auto &tex2D : tex2Ds)
            {
                auto tex2DStrSize = tex2D->serializeSize();
                size += sizeof(tex2DStrSize);
                size += tex2DStrSize;
            }

            auto tex3DsCount {tex3Ds.size()};
            size += sizeof(tex3DsCount);

            for (const auto &tex3D : tex3Ds) {
                auto& key = tex3D.first;
                auto& value = tex3D.second;

                auto key_len = key.length();
                auto value_len = value.length();

                size += sizeof(key_len);
                size += key.length();
                size += sizeof(value_len);
                size += value.length();
            }

            auto transformLen{transform.size()};
            size += sizeof(transformLen);
            size += transformLen;

            return size;
        }

        void serialize(char *str) const
        {
            size_t i{0};

            auto mtlidkeyLen{mtlidkey.size()};
            memcpy(str + i, &mtlidkeyLen, sizeof(mtlidkeyLen));
            i += sizeof(mtlidkeyLen);

            mtlidkey.copy(str + i, mtlidkeyLen);
            i += mtlidkeyLen;

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

            auto tex3DsCount {tex3Ds.size()};
            memcpy(str + i, &tex3DsCount, sizeof(tex3DsCount));
            i += sizeof(tex3DsCount);
            
            for (const auto &tex3D : tex3Ds) {
                auto& key = tex3D.first;
                auto& value = tex3D.second;

                auto key_len = key.length();
                auto value_len = value.length();

                memcpy(str + i, &key_len, sizeof(key_len));
                i += sizeof(key_len);
                memcpy(str + i, key.c_str(), key_len);
                i += key_len;

                memcpy(str + i, &value_len, sizeof(value_len));
                i += sizeof(value_len);
                memcpy(str + i, value.c_str(), value_len);
                i += value_len;
            }

            auto transformLen{transform.size()};
            memcpy(str + i, &transformLen, sizeof(transformLen));
            i += sizeof(transformLen);

            transform.copy(str + i, transformLen);
            i += transformLen;
        }

        std::vector<char> serialize() const
        {
            std::vector<char> str(serializeSize());
            serialize(str.data());
            return str;
        }

        void deserialize(const char *str)
        {
            size_t i{0};

            size_t mtlidkeyLen;
            memcpy(&mtlidkeyLen, str + i, sizeof(mtlidkeyLen));
            i += sizeof(mtlidkeyLen);

            this->mtlidkey = std::string{str + i, mtlidkeyLen};
            i += mtlidkeyLen;

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


            size_t tex3DsCount;
            memcpy(&tex3DsCount, str + i, sizeof(tex3DsCount));
            i += sizeof(tex3DsCount);
            this->tex3Ds.resize(tex3DsCount);
            
            for (size_t j=0; j < tex3DsCount; ++j) 
            {
                size_t key_len;
                memcpy(&key_len, str + i, sizeof(key_len));
                i += sizeof(key_len);
                //std::vector<char> key_str;
                //key_str.resize(key_len);
                //memcpy(key_str.data(), str + i, key_len);
                auto key = std::string{str + i, key_len};
                i += key_len; 

                size_t value_len;
                memcpy(&value_len, str + i, sizeof(value_len));
                i += sizeof(value_len);
                //std::vector<char> value_str;
                //value_str.resize(value_len);
                //memcpy(value_str.data(), str + i, value_len);
                auto value = std::string{str + i, value_len};
                i += value_len; 

                this->tex3Ds[j] = std::make_pair(key, value);;
            }

            size_t transformLen;
            memcpy(&transformLen, str + i, sizeof(transformLen));
            i += sizeof(transformLen);

            this->transform = std::string{str + i, transformLen};
            i += transformLen;
        }

        static MaterialObject deserialize(const std::vector<char> &str)
        {
            MaterialObject mtl;
            mtl.deserialize(str.data());
            return mtl;
        }

    }; // struct Material

} // namespace zeno
