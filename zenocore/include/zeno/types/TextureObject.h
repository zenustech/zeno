#pragma once

#include <zeno/core/IObject.h>
#include <string>
#include <cstring>
#include <vector>

namespace zeno
{

    struct Texture2DObject
        : IObjectClone<Texture2DObject>
    {
        std::string path;

        enum class TexWrapEnum
        {
            REPEAT,
            MIRRORED_REPEAT,
            CLAMP_TO_EDGE,
            CLAMP_TO_BORDER,
        };
        TexWrapEnum wrapS;
        TexWrapEnum wrapT;

        enum class TexFilterEnum
        {
            NEAREST,
            LINEAR,
            NEAREST_MIPMAP_NEAREST,
            LINEAR_MIPMAP_NEAREST,
            NEAREST_MIPMAP_LINEAR,
            LINEAR_MIPMAP_LINEAR,
        };
        TexFilterEnum minFilter;
        TexFilterEnum magFilter;

        size_t serializeSize()
        {
            size_t size{0};

            auto pathLen{path.size()};
            size += sizeof(pathLen);
            size += pathLen;

            size += sizeof(wrapS);
            size += sizeof(wrapT);

            size += sizeof(minFilter);
            size += sizeof(magFilter);

            return size;
        }

        std::vector<char> serialize()
        {
            std::vector<char> str;
            str.resize(serializeSize());

            size_t i{0};

            auto pathLen{path.size()};
            memcpy(str.data() + i, &pathLen, sizeof(pathLen));
            i += sizeof(pathLen);
            path.copy(str.data() + i, pathLen);
            i += pathLen;

            memcpy(str.data() + i, &wrapS, sizeof(wrapS));
            i += sizeof(wrapS);
            memcpy(str.data() + i, &wrapT, sizeof(wrapT));
            i += sizeof(wrapT);

            memcpy(str.data() + i, &minFilter, sizeof(minFilter));
            i += sizeof(minFilter);
            memcpy(str.data() + i, &magFilter, sizeof(magFilter));
            i += sizeof(magFilter);

            return str;
        }
        
        static Texture2DObject deserialize(const std::vector<char> &str)
        {
            Texture2DObject tex;

            size_t i{0};

            size_t pathLen;
            memcpy(&pathLen, str.data() + i, sizeof(pathLen));
            i += sizeof(pathLen);

            tex.path = std::string{str.data() + i, pathLen};
            i += pathLen;

            memcpy(&(tex.wrapS), str.data() + i, sizeof(wrapS));
            i += sizeof(wrapS);

            memcpy(&(tex.wrapT), str.data() + i, sizeof(wrapT));
            i += sizeof(wrapT);

            memcpy(&(tex.minFilter), str.data() + i, sizeof(minFilter));
            i += sizeof(minFilter);

            memcpy(&(tex.magFilter), str.data() + i, sizeof(magFilter));
            i += sizeof(magFilter);

            return tex;
        }

    }; // struct Texture


    struct TextureObjectVDB: IObjectClone<TextureObjectVDB>
    {
        std::string path;
        std::string channel;

        enum struct ElementType {
            Fp32, Fp16, Fp8, Fp4
        };
        ElementType eleType;

        size_t serializeSize()
        {
            size_t size{0};

            auto pathLen{path.size()};
            size += sizeof(pathLen);
            size += pathLen;

            auto channelLen(channel.size());
            size += sizeof(channelLen);
            size += channelLen;

            size += sizeof(eleType);

            return size;
        }

        std::vector<char> serialize()
        {
            std::vector<char> str;
            str.resize(serializeSize());

            size_t i{0}; 
            {
                auto pathLen{path.size()};
                memcpy(str.data() + i, &pathLen, sizeof(pathLen));
                i += sizeof(pathLen);
                path.copy(str.data() + i, pathLen);
                i += pathLen;
            }

            {
                auto channelLen(channel.size());
                memcpy(str.data() + i, &channelLen, sizeof(channelLen));
                i += sizeof(channelLen);
                channel.copy(str.data() + i, channelLen);
                i += channelLen;
            }

            {
                memcpy(str.data() + i, &eleType, sizeof(eleType));
                i += sizeof(eleType);
            }

            return str;
        }
        
        static TextureObjectVDB deserialize(const std::vector<char> &str)
        {
            TextureObjectVDB tex;
            size_t i{0}; 

            {
                size_t pathLen;
                memcpy(&pathLen, str.data() + i, sizeof(pathLen));
                i += sizeof(pathLen);

                tex.path = std::string{str.data() + i, pathLen};
                i += pathLen;
            }

            {
                size_t channelLen;
                memcpy(&channelLen, str.data() + i, sizeof(channelLen));
                i += sizeof(channelLen);

                tex.channel = std::string{str.data() + i, channelLen};
                i += channelLen;
            }

            {
                memcpy(&(tex.eleType), str.data() + i, sizeof(eleType));
                i += sizeof(eleType);
            }

            return tex;
        }

    }; // struct Texture

} // namespace zeno
