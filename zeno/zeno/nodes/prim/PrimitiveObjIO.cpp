#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/utils/vec.h>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <cstdio>

namespace {

struct hash_vec3i {
    size_t operator()(const zeno::vec3i& rhs) const {
        return std::hash<int32_t>()(rhs[0] + rhs[1] + rhs[2]);
    }
};

struct cmp_vec3i {
    bool operator()(const zeno::vec3i& lhs, const zeno::vec3i& rhs) const {
        return lhs[0] == rhs[0] && lhs[1] == rhs[1] && lhs[2] == rhs[2];
    }
};

static void readobj(
        std::vector<zeno::vec3f> &vertices,
        std::vector<zeno::vec3f> &nrms,
        std::vector<zeno::vec3f> &uvs,
        std::vector<zeno::vec3i> &indices,
        const char *path)
{
    FILE *fp = fopen(path, "r");
    if (!fp) {
        perror(path);
        abort();
    }

    //printf("o %s\n", path);

    char hdr[128];
    while (EOF != fscanf(fp, "%s", hdr)) 
    {
        if (!strcmp(hdr, "vn")) {
            zeno::vec3f normal;
            fscanf(fp, "%f %f %f\n", &normal[0], &normal[1], &normal[2]);
            nrms.push_back(normal);

        } else if (!strcmp(hdr, "vt")) {
            zeno::vec3f uv;
            fscanf(fp, "%f %f %f\n", &uv[0], &uv[1], &uv[2]);
            uvs.push_back(uv);
        } else if (!strcmp(hdr, "v")) {
            zeno::vec3f vertex;
            fscanf(fp, "%f %f %f\n", &vertex[0], &vertex[1], &vertex[2]);
            //printf("v %f %f %f\n", vertex[0], vertex[1], vertex[2]);
            vertices.push_back(vertex);
        
        } else if (!strcmp(hdr, "f")) {
            // face vert/uv/normal
            zeno::vec3i last_index, first_index, index;

            fscanf(fp, "%d/%d/%d", &index[0], &index[1], &index[2]);
            first_index = index;

            fscanf(fp, "%d/%d/%d", &index[0], &index[1], &index[2]);
            last_index = index;

            while (fscanf(fp, "%d/%d/%d", &index[0], &index[1], &index[2]) > 0) {
                indices.push_back(first_index - 1);
                indices.push_back(last_index - 1);
                indices.push_back(index - 1);

                last_index = index;
            }
        }
    }
    std::vector<zeno::vec3f> _outvertices;
    std::vector<zeno::vec3f> _outnormals;
    std::vector<zeno::vec3f> _outtexs;
    std::vector<zeno::vec3i> _outindices;

    std::unordered_map<zeno::vec3i, int, hash_vec3i, cmp_vec3i> reindex;
    for (auto index : indices) {
        if (reindex.count(index) == 0) {
            int new_index = reindex.size();
            reindex[index] = new_index;
            _outvertices.push_back(vertices[index[0]]);
            _outtexs.push_back(uvs[index[1]]);
            _outnormals.push_back(nrms[index[2]]);
        }
    }
    for (int i = 0; i < indices.size(); i++) {
        if (i % 3 == 0) {
            auto index = zeno::vec3i(
                reindex[indices[i]],
                reindex[indices[i + 1]],
                reindex[indices[i + 2]]
            );
            _outindices.push_back(index);
        }
    }
    vertices = std::move(_outvertices);
    nrms = std::move(_outnormals);
    uvs = std::move(_outtexs);
    indices = std::move(_outindices);
    fclose(fp);
}


struct ReadObjPrimitive : zeno::INode {
    virtual void apply() override {
        auto path = get_input<zeno::StringObject>("path")->get();
        auto prim = std::make_shared<zeno::PrimitiveObject>();
        auto &pos = prim->add_attr<zeno::vec3f>("pos");
        auto &nrm = prim->add_attr<zeno::vec3f>("nrm");
        auto &uv = prim->add_attr<zeno::vec3f>("uv");
        readobj(pos, nrm, uv, prim->tris, path.c_str());
        prim->resize(pos.size());
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(ReadObjPrimitive,
        { /* inputs: */ {
        {"readpath", "path"},
        }, /* outputs: */ {
        "prim",
        }, /* params: */ {
        }, /* category: */ {
        "primitive",
        }});

struct ImportObjPrimitive : ReadObjPrimitive {
};

ZENDEFNODE(ImportObjPrimitive,
        { /* inputs: */ {
        {"readpath", "path"},
        }, /* outputs: */ {
        "prim",
        }, /* params: */ {
        }, /* category: */ {
        "primitive",
        }});



static void writeobj(
        std::vector<zeno::vec3f> const &vertices,
        std::vector<zeno::vec3i> const &indices,
        const char *path)
{
    FILE *fp = fopen(path, "w");
    if (!fp) {
        perror(path);
        abort();
    }

    for (auto const &vert: vertices) {
        fprintf(fp, "v %f %f %f\n", vert[0], vert[1], vert[2]);
    }

    for (auto const &ind: indices) {
        fprintf(fp, "f %d %d %d\n", ind[0] + 1, ind[1] + 1, ind[2] + 1);
    }
    fclose(fp);
}


struct WriteObjPrimitive : zeno::INode {
    virtual void apply() override {
        auto path = get_input<zeno::StringObject>("path")->get();
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto &pos = prim->attr<zeno::vec3f>("pos");
        writeobj(pos, prim->tris, path.c_str());
    }
};

ZENDEFNODE(WriteObjPrimitive,
        { /* inputs: */ {
        {"writepath", "path"},
        "prim",
        }, /* outputs: */ {
        }, /* params: */ {
        }, /* category: */ {
        "primitive",
        }});

struct ExportObjPrimitive : WriteObjPrimitive {
    virtual void apply() override {
        auto path = get_input<zeno::StringObject>("path")->get();
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto &pos = prim->attr<zeno::vec3f>("pos");
        writeobj(pos, prim->tris, path.c_str());
    }
};

ZENDEFNODE(ExportObjPrimitive,
        { /* inputs: */ {
        {"writepath", "path"},
        "prim",
        }, /* outputs: */ {
        }, /* params: */ {
        }, /* category: */ {
        "primitive",
        }});

}
