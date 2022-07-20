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
        std::vector<float> ufloat;
        std::vector<zeno::vec2f> uv2f;
        std::vector<zeno::vec3f> uv3f;
        std::vector<zeno::vec4f> uv4f;

        size_t serializeSize()
        {
            size_t size{0};

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
            
            auto ufloatSize{ufloat.size()};
            size += sizeof(ufloatSize);
            for (const auto tmp : ufloat)
            {
                size += sizeof(tmp);
            }

            // auto uv2fSize{uv2f.size()};
            // size += sizeof(uv2fSize);
            // for (const auto tmp : uv2f)
            // {
            //     size_t subSize{0};
            //     subSize += sizeof(tmp[0]);
            //     subSize += sizeof(tmp[1]);

            //     auto tmpSize = subSize;
            //     size += sizeof(tmpSize);
            //     size += tmpSize;
            // }

            // auto uv3fSize{uv3f.size()};
            // size += sizeof(uv3fSize);
            // for (const auto tmp : uv3f)
            // {
            //     size_t subSize{0};
            //     subSize += sizeof(tmp[0]);
            //     subSize += sizeof(tmp[1]);
            //     subSize += sizeof(tmp[2]);

            //     auto tmpSize = subSize;
            //     size += sizeof(tmpSize);
            //     size += tmpSize;
            // }

            // auto uv4fSize{uv4f.size()};
            // size += sizeof(uv4fSize);
            // for (const auto tmp : uv4f)
            // {
            //     size_t subSize{0};
            //     subSize += sizeof(tmp[0]);
            //     subSize += sizeof(tmp[1]);
            //     subSize += sizeof(tmp[2]);
            //     subSize += sizeof(tmp[3]);

            //     auto tmpSize = subSize;
            //     size += sizeof(tmpSize);
            //     size += tmpSize;
            // }

            return size;
        }

        std::vector<char> serialize()
        {
            std::vector<char> str;
            str.resize(serializeSize());

            size_t i{0};

            auto vertLen{vert.size()};
            memcpy(str.data() + i, &vertLen, sizeof(vertLen));
            i += sizeof(vertLen);

            vert.copy(str.data() + i, vertLen);
            i += vertLen;

            auto fragLen{frag.size()};
            memcpy(str.data() + i, &fragLen, sizeof(fragLen));
            i += sizeof(fragLen);

            frag.copy(str.data() + i, fragLen);
            i += fragLen;

            auto commonLen{common.size()};
            memcpy(str.data() + i, &commonLen, sizeof(commonLen));
            i += sizeof(commonLen);

            common.copy(str.data() + i, commonLen);
            i += commonLen;

            auto extensionsLen{extensions.size()};
            memcpy(str.data() + i, &extensionsLen, sizeof(extensionsLen));
            i += sizeof(extensionsLen);

            extensions.copy(str.data() + i, extensionsLen);
            i += extensionsLen;

            auto tex2DsSize{tex2Ds.size()};
            memcpy(str.data() + i, &tex2DsSize, sizeof(tex2DsSize));
            i += sizeof(tex2DsSize);

            for (const auto &tex2D : tex2Ds)
            {
                auto tex2DStr = tex2D->serialize();
                auto tex2DStrSize = tex2DStr.size();
                memcpy(str.data() + i, &tex2DStrSize, sizeof(tex2DStrSize));
                i += sizeof(tex2DStrSize);

                memcpy(str.data() + i, tex2DStr.data(), tex2DStrSize);
                i += tex2DStrSize;
            }

            auto ufloatSize{ufloat.size()};
            memcpy(str.data() + i, &ufloatSize, sizeof(ufloatSize));
            i += sizeof(ufloatSize);
            for (const auto tmp : ufloat)
            {
                memcpy(str.data() + i, &tmp, sizeof(tmp));
                i += sizeof(tmp);
            }
            
            return str;
        }

        static MaterialObject deserialize(const std::vector<char> &str)
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

            size_t tex2DsSize;
            memcpy(&tex2DsSize, str.data() + i, sizeof(tex2DsSize));
            i += sizeof(tex2DsSize);
            mtl.tex2Ds.resize(tex2DsSize);

            for (size_t j{0}; j < tex2DsSize; ++j)
            {
                size_t tex2DStrSize;
                memcpy(&tex2DStrSize, str.data() + i, sizeof(tex2DStrSize));
                i += sizeof(tex2DStrSize);

                std::vector<char> tex2DStr;
                tex2DStr.resize(tex2DStrSize);
                memcpy(tex2DStr.data(), str.data() + i, tex2DStrSize);
                i += tex2DStrSize;

                auto tex2D = std::make_shared<Texture2DObject>(
                    Texture2DObject::deserialize(tex2DStr));
                mtl.tex2Ds[j] = tex2D;
            }

            size_t ufloatSize;
            memcpy(&ufloatSize, str.data() + i, sizeof(ufloatSize));
            i += sizeof(ufloatSize);
            mtl.ufloat.resize(ufloatSize);
            for (size_t j{0}; j < ufloatSize; ++j)
            {
                float tmpData;
                memcpy(&tmpData, str.data() + i, sizeof(tmpData));
                i += sizeof(tmpData);
            }

            return mtl;
        }

    }; // struct Material

} // namespace zeno
