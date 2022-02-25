#pragma once

#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/MaterialObject.h>
#include <zeno/utils/vec.h>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <cstdio>


namespace zeno {


static void writezpm(PrimitiveObject const *prim, const char *path) {
    FILE *fp = fopen(path, "wb");

    char signature[9] = "\x7fZPMv001";
    fwrite(signature, sizeof(char), 8, fp);

    size_t size = prim->size();
    fwrite(&size, sizeof(size), 1, fp);

    int count = prim->num_attrs();
    fwrite(&count, sizeof(count), 1, fp);

    prim->foreach_attr([&] (auto const &name, auto const &arr) {
        char type[5];
        memset(type, 0, sizeof(type));
        if constexpr (0) {
#define _PER_ALTER(T, id) \
        } else if constexpr (std::is_same_v<std::decay_t<decltype(arr[0])>, T>) { \
            strcpy(type, id);
        _PER_ALTER(float, "f")
        _PER_ALTER(zeno::vec3f, "3f")
#undef _PER_ALTER
        } else {
            //printf("%s\n", name);
            assert(0 && "Bad primitive variant type");
        }
        fwrite(type, 4, 1, fp);

        size_t namelen = name.size();
        fwrite(&namelen, sizeof(namelen), 1, fp);
        fwrite(name.c_str(), sizeof(name[0]), namelen, fp);
    });

    prim->foreach_attr([&] (auto const &key, auto const &attr) {
        assert(attr.size() == size);
        fwrite(attr.data(), sizeof(attr[0]), size, fp);
    });

    size = prim->points.size();
    fwrite(&size, sizeof(size_t), 1, fp);
    fwrite(prim->points.data(), sizeof(prim->points[0]), prim->points.size(), fp);

    size = prim->lines.size();
    fwrite(&size, sizeof(size_t), 1, fp);
    fwrite(prim->lines.data(), sizeof(prim->lines[0]), prim->lines.size(), fp);

    size = prim->tris.size();
    fwrite(&size, sizeof(size_t), 1, fp);
    fwrite(prim->tris.data(), sizeof(prim->tris[0]), prim->tris.size(), fp);

    size = prim->quads.size();
    fwrite(&size, sizeof(size_t), 1, fp);
    fwrite(prim->quads.data(), sizeof(prim->quads[0]), prim->quads.size(), fp);

    if (prim->mtl != nullptr)
    {
        auto mtlStr = prim->mtl->serialize();
        size = mtlStr.size();
        fwrite(&size, sizeof(size_t), 1, fp);
        fwrite(mtlStr.data(), sizeof(mtlStr[0]), mtlStr.size(), fp);
    }
    else
    {
        size = 0;
        fwrite(&size, sizeof(size_t), 1, fp);
    }

    fclose(fp);
}

static void readzpm(PrimitiveObject *prim, const char *path) {
    FILE *fp = fopen(path, "rb");

    char signature[9] = "";
    fread(signature, sizeof(char), 8, fp);
    signature[8] = 0;
    assert(!strcmp(signature, "\x7fZPMv001"));

    size_t size = 0;
    fread(&size, sizeof(size), 1, fp);
    //printf("size = %zd\n", size);
    prim->resize(size);

    int count = 0;
    fread(&count, sizeof(count), 1, fp);
    //printf("count = %d\n", count);
    assert(count < 1024);

    for (int i = 0; i < count; i++) {
        //printf("parsing attr %d\n", i);

        char type[5];
        fread(type, 4, 1, fp);
        type[4] = '\0';

        size_t namelen = 0;
        fread(&namelen, sizeof(namelen), 1, fp);
        //printf("attr namelen = %zd\n", namelen);
        assert(namelen < 1024);
        char *namebuf = (char *)alloca(namelen + 1);
        fread(namebuf, sizeof(namebuf[0]), namelen, fp);
        namebuf[namelen] = '\0';
        std::string name(namebuf);

        //printf("attr `%s` of type `%s`\n", namebuf, type);

        if (0) {
#define _PER_ALTER(T, id) \
        } else if (!strcmp(type, id)) { \
            prim->add_attr<T>(name);
        _PER_ALTER(float, "f")
        _PER_ALTER(zeno::vec3f, "3f")
#undef _PER_ALTER
        } else {
            //printf("%s\n", name);
            assert(0 && "Bad primitive variant type");
        }
    }
    assert(prim->num_attrs() == count);

    // assuming prim->m_attrs is an ordered map
    prim->foreach_attr([&] (auto const &key, auto &attr) {
        //printf("reading array of attr `%s`\n", key.c_str());

        assert(attr.size() == size);
        fread(attr.data(), sizeof(attr[0]), size, fp);
    });

    fread(&size, sizeof(size_t), 1, fp);
    prim->points.resize(size);
    fread(prim->points.data(), sizeof(prim->points[0]), prim->points.size(), fp);

    fread(&size, sizeof(size_t), 1, fp);
    prim->lines.resize(size);
    fread(prim->lines.data(), sizeof(prim->lines[0]), prim->lines.size(), fp);

    fread(&size, sizeof(size_t), 1, fp);
    prim->tris.resize(size);
    fread(prim->tris.data(), sizeof(prim->tris[0]), prim->tris.size(), fp);

    fread(&size, sizeof(size_t), 1, fp);
    prim->quads.resize(size);
    fread(prim->quads.data(), sizeof(prim->quads[0]), prim->quads.size(), fp);

    fread(&size, sizeof(size_t), 1, fp);
    if (size != 0)
    {
        std::vector<char> mtlStr;
        mtlStr.resize(size);
        fread(mtlStr.data(), sizeof(mtlStr[0]), mtlStr.size(), fp);
        prim->mtl = std::make_shared<MaterialObject>(MaterialObject::deserialize(mtlStr));
    }

    fclose(fp);
}


}
