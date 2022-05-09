#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/DictObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/PrimitiveTools.h>
#include <zeno/utils/string.h>
#include <zeno/utils/logger.h>
#include <zeno/utils/vec.h>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <fstream>


namespace zeno {


static zeno::vec3i read_index(std::string str) {
    zeno::vec3i face(0, 0, 0);
    auto items = zeno::split_str(str, '/');
    for (auto i = 0; i < items.size(); i++) {
        if (items[i].empty()) {
            continue;
        }
        face[i] = std::stoi(items[i]);
    }
    return face - 1;

}
static zeno::vec3f read_vec3f(std::vector<std::string> items) {
    zeno::vec3f vec(0, 0, 0);
    int i = 0;
    for (auto item: items) {
        if (item.size() != 0) {
            vec[i] = std::stof(item);
            i += 1;
        }
    }
    return vec;
}

void read_obj_file(
        std::vector<zeno::vec3f> &vertices,
        std::vector<zeno::vec3f> &uvs,
        std::vector<zeno::vec3f> &normals,
        std::vector<zeno::vec3i> &indices,
        //std::vector<zeno::vec3i> &uv_indices,
        //std::vector<zeno::vec3i> &normal_indices,
        const char *path)
{


    auto is = std::ifstream(path);
    while (!is.eof()) {
        std::string line;
        std::getline(is, line);
        line = zeno::trim_string(line);
        if (line.empty()) {
            continue;
        }
        auto items = zeno::split_str(line, ' ');
        items.erase(items.begin());

        if (zeno::starts_with(line, "v ")) {
            vertices.push_back(read_vec3f(items));

        } else if (zeno::starts_with(line, "vt ")) {
            uvs.push_back(read_vec3f(items));

        } else if (zeno::starts_with(line, "vn ")) {
            normals.push_back(read_vec3f(items));

        } else if (zeno::starts_with(line, "f ")) {
            zeno::vec3i first_index = read_index(items[0]);
            zeno::vec3i last_index = read_index(items[1]);

            for (auto i = 2; i < items.size(); i++) {
                zeno::vec3i index = read_index(items[i]);
                zeno::vec3i face(first_index[0], last_index[0], index[0]);
                //zeno::vec3i face_uv(first_index[1], last_index[1], index[1]);
                //zeno::vec3i face_normal(first_index[2], last_index[2], index[2]);
                indices.push_back(face);
                //uv_indices.push_back(face_uv);
                //normal_indices.push_back(face_normal);
                last_index = index;
            }
        }
    }
}


struct ReadObjPrimitive : zeno::INode {
    virtual void apply() override {
        auto path = get_input<zeno::StringObject>("path")->get();
        auto prim = std::make_shared<zeno::PrimitiveObject>();
        auto &pos = prim->verts;
        auto &uv = prim->verts.add_attr<zeno::vec3f>("uv");
        auto &norm = prim->verts.add_attr<zeno::vec3f>("nrm");
        auto &tris = prim->tris;
        //auto &triuv = prim->tris.add_attr<zeno::vec3i>("uv");
        //auto &trinorm = prim->tris.add_attr<zeno::vec3i>("nrm");
        read_obj_file(pos, uv, norm, tris, /*triuv, trinorm,*/ path.c_str());
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
        std::shared_ptr<zeno::PrimitiveObject> &prim,
        const char *path)
{
    FILE *fp = fopen(path, "w");
    if (!fp) {
        perror(path);
        abort();
    }
    for (auto const &vert: prim->verts) {
        fprintf(fp, "v %f %f %f\n", vert[0], vert[1], vert[2]);
    }
    if (prim->tris.has_attr("uv0")) {
        auto& uv0 = prim->tris.attr<zeno::vec3f>("uv0");
        auto& uv1 = prim->tris.attr<zeno::vec3f>("uv1");
        auto& uv2 = prim->tris.attr<zeno::vec3f>("uv2");
        for (auto i = 0; i < prim->tris.size(); i++) {
            fprintf(fp, "vt %f %f %f\n", uv0[i][0], uv0[i][1], uv0[i][2]);
            fprintf(fp, "vt %f %f %f\n", uv1[i][0], uv1[i][1], uv1[i][2]);
            fprintf(fp, "vt %f %f %f\n", uv2[i][0], uv2[i][1], uv2[i][2]);
        }
        int count = 0;
        for (auto const &ind: prim->tris) {
            fprintf(fp, "f %d/%d %d/%d %d/%d\n",
                ind[0] + 1, count + 1,
                ind[1] + 1, count + 2,
                ind[2] + 1, count + 3
            );
            count += 3;
        }
    } else {
        for (auto const &ind: prim->tris) {
            fprintf(fp, "f %d %d %d\n", ind[0] + 1, ind[1] + 1, ind[2] + 1);
        }
    }
    fclose(fp);
}


struct WriteObjPrimitive : zeno::INode {
    virtual void apply() override {
        auto path = get_input<zeno::StringObject>("path")->get();
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto &pos = prim->attr<zeno::vec3f>("pos");
        writeobj(prim, path.c_str());
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
        writeobj(prim, path.c_str());
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

//--------------------- dict--------------------------//
std::shared_ptr<zeno::DictObject>
read_obj_file_dict(
        std::vector<zeno::vec3f> &vertices,
        //std::vector<zeno::vec3f> &uvs,
        //std::vector<zeno::vec3f> &normals,
        std::vector<zeno::vec3i> &indices,
        //std::vector<zeno::vec3i> &uv_indices,
        //std::vector<zeno::vec3i> &normal_indices,
        const char *path)
{
    std::map<std::string, std::vector<zeno::vec3i>> dict;
    std::string sub_name = "unnamed";
    std::vector<zeno::vec3i> sub_indices;


    std::shared_ptr<zeno::DictObject> prims = std::make_shared<zeno::DictObject>();

    size_t vert_offset = 0;
    size_t pre_vert_offset = 0;
    auto is = std::ifstream(path);
    bool has_read_o_tag = false;

    while (!is.eof()) {
        std::string line;
        std::getline(is, line);
        line = zeno::trim_string(line);
        if (line.empty()) {
            continue;
        }
        auto items = zeno::split_str(line, ' ');
        items.erase(items.begin());

        if (zeno::starts_with(line, "v ")) {

            // std::cout << "PARSING V" << std::endl;
            vertices.push_back(read_vec3f(items));
            vert_offset += 1;
        /*
        } else if (zeno::starts_with(line, "vt ")) {
            uvs.push_back(read_vec3f(items));
        } else if (zeno::starts_with(line, "vn ")) {
            normals.push_back(read_vec3f(items));
        */
        } else if (zeno::starts_with(line, "f ")) {

            // std::cout << "PARSING F" << std::endl;

            zeno::vec3i first_index = read_index(items[0]);
            zeno::vec3i last_index = read_index(items[1]);

            for (auto i = 2; i < items.size(); i++) {
                zeno::vec3i index = read_index(items[i]);
                // only work for .obj file with sub objects
                zeno::vec3i face(first_index[0], last_index[0], index[0]);
                //zeno::vec3i face_uv(first_index[1], last_index[1], index[1]);
                //zeno::vec3i face_normal(first_index[2], last_index[2], index[2]);
                indices.push_back(face);
                sub_indices.push_back(face);
                //uv_indices.push_back(face_uv);
                //normal_indices.push_back(face_normal);
                last_index = index;
            }
        } else if (zeno::starts_with(line, "o ")) {
            // if we have already parse the o tag, the subname, vertices and faces data should have already been read
            if(has_read_o_tag){
                auto sub_prim = std::make_shared<zeno::PrimitiveObject>();
                sub_prim->tris = sub_indices;
                for(size_t i = 0;i < sub_prim->tris.size();++i){
                    sub_prim->tris[i] -= zeno::vec3i(pre_vert_offset);
                }
                sub_prim->verts = std::vector(vertices.begin() + pre_vert_offset,vertices.end()- 0);
                std::vector<zeno::vec3f>(&vertices[pre_vert_offset],&vertices[vert_offset]);
                prims->lut[sub_name] = sub_prim;
            }
            // Update the sub_obj name
            sub_name = items[0];
            has_read_o_tag = true;
            // update the vertex offset for seperating the current obj's vertices from the previous one's
            pre_vert_offset = vert_offset;
            sub_indices.clear();

            zeno::log_debug("sub_mesh: {}\n", sub_name);

        }
    }
    // if there is no sub objects, output the mesh as a whole unameed subobject
    auto sub_prim = std::make_shared<zeno::PrimitiveObject>();
    if(!has_read_o_tag){
        sub_prim->verts = vertices;
        sub_prim->tris = indices;
    }else{
        zeno::log_debug("sub_mesh: {}\n", sub_name);
        sub_prim->verts = std::vector(vertices.begin() + pre_vert_offset,vertices.end() - 0);
        sub_prim->tris = sub_indices;
        for(size_t i = 0;i < sub_prim->tris.size();++i){
            sub_prim->tris[i] -= zeno::vec3i(pre_vert_offset);
        }
    }
    prims->lut[sub_name] = sub_prim;
    return prims;
}
struct ReadObjPrimitiveDict : zeno::INode {
    virtual void apply() override {
        auto path = get_input<zeno::StringObject>("path")->get();
        auto prim = std::make_shared<zeno::PrimitiveObject>();
        auto &pos = prim->verts;
        //auto &uv = prim->verts.add_attr<zeno::vec3f>("uv");
        //auto &norm = prim->verts.add_attr<zeno::vec3f>("nrm");
        auto &tris = prim->tris;
        //auto &triuv = prim->tris.add_attr<zeno::vec3i>("uv");
        //auto &trinorm = prim->tris.add_attr<zeno::vec3i>("nrm");
        auto prims = read_obj_file_dict(pos, /*uv, norm,*/ tris, /*triuv, trinorm,*/ path.c_str());
        set_output("prim", std::move(prim));
        set_output("dict", std::move(prims));
    }
};

ZENDEFNODE(ReadObjPrimitiveDict,
        { /* inputs: */ {
        {"readpath", "path"},
        }, /* outputs: */ {
        "prim",
        "dict",
        }, /* params: */ {
        }, /* category: */ {
        "primitive",
        }});

}
