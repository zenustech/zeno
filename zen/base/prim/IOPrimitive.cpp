#include <zen/zen.h>
#include <zen/PrimitiveObject.h>
#include <zen/StringObject.h>
#include <glm/glm.hpp>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <cstdio>

namespace zenbase {


struct ExportPrimitive : zen::INode {
  virtual void apply() override {
    auto path = get_input("path")->as<StringObject>();
    auto prim = get_input("prim")->as<PrimitiveObject>();

    FILE *fp = fopen(path->get().c_str(), "wb");

    char signature[9] = "\x7fZPMv001";
    fwrite(signature, sizeof(char), 8, fp);

    size_t size = prim->size();
    fwrite(&size, sizeof(size), 1, fp);

    int count = prim->m_attrs.size();
    fwrite(&count, sizeof(count), 1, fp);

    for (auto const &[name, _]: prim->m_attrs) {
        char type[5];
        if (0) {
#define _PER_ALTER(T, id) \
        } else if (prim->attr_is<T>(name)) { \
            strcpy(type, id);
        _PER_ALTER(float, "f")
        _PER_ALTER(glm::vec3, "f3")
#undef _PER_ALTER
        } else {
            printf("%s\n", name);
            assert(0 && "Bad primitive variant type");
        }
        fwrite(type, 4, 1, fp);

        size_t namelen = name.size();
        fwrite(&namelen, sizeof(namelen), 1, fp);
        fwrite(name.c_str(), sizeof(name[0]), namelen, fp);
    }

    for (auto const &[key, _]: prim->m_attrs) {
        std::visit([=](auto const &attr) {
            assert(attr.size() == size);
            fwrite(attr.data(), sizeof(attr[0]), size, fp);
        }, prim->attr(key));
    }

    // TODO: export triangles too.

    fclose(fp);
  }
};

static int defExportPrimitive = zen::defNodeClass<ExportPrimitive>("ExportPrimitive",
    { /* inputs: */ {
    "prim",
    "path",
    }, /* outputs: */ {
    }, /* params: */ {
    }, /* category: */ {
    "primitive",
    }});


struct ImportPrimitive : zen::INode {
  virtual void apply() override {
    auto path = get_input("path")->as<StringObject>();
    auto prim = zen::IObject::make<PrimitiveObject>();

    FILE *fp = fopen(path->get().c_str(), "rb");

    char signature[9] = "";
    fread(signature, sizeof(char), 8, fp);
    signature[8] = 0;
    assert(!strcmp(signature, "\x7fZPMv001"));

    size_t size = 0;
    fread(&size, sizeof(size), 1, fp);
    printf("size = %zd\n", size);
    prim->resize(size);

    int count = 0;
    fread(&count, sizeof(count), 1, fp);
    printf("count = %d\n", count);
    assert(count < 1024);

    for (int i = 0; i < count; i++) {
        printf("parsing attr %d\n", i);

        char type[5];
        fread(type, 4, 1, fp);
        type[4] = '\0';

        size_t namelen = 0;
        fread(&namelen, sizeof(namelen), 1, fp);
        printf("attr namelen = %zd\n", namelen);
        assert(namelen < 1024);
        char namebuf[namelen + 1];
        fread(namebuf, sizeof(namebuf[0]), namelen, fp);
        namebuf[namelen] = '\0';
        std::string name(namebuf);

        printf("attr `%s` of type `%s`\n", namebuf, type);

        if (0) {
#define _PER_ALTER(T, id) \
        } else if (!strcmp(type, id)) { \
            prim->add_attr<T>(name);
        _PER_ALTER(float, "f")
        _PER_ALTER(glm::vec3, "f3")
#undef _PER_ALTER
        } else {
            printf("%s\n", name);
            assert(0 && "Bad primitive variant type");
        }
    }
    assert(prim->m_attrs.size() == count);

    // assuming prim->m_attrs is an ordered map
    for (auto const &[key, _]: prim->m_attrs) {
        printf("reading array of attr `%s`\n", key.c_str());

        std::visit([=](auto &attr) {
            assert(attr.size() == size);
            fread(attr.data(), sizeof(attr[0]), size, fp);
        }, prim->attr(key));
    }

    fclose(fp);

    set_output("prim", prim);
  }
};

static int defImportPrimitive = zen::defNodeClass<ImportPrimitive>("ImportPrimitive",
    { /* inputs: */ {
    "path",
    }, /* outputs: */ {
    "prim",
    }, /* params: */ {
    }, /* category: */ {
    "primitive",
    }});

}
