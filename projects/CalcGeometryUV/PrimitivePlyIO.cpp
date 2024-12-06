#include <cmath>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/utils/string.h>
#include <zeno/utils/vec.h>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <fstream>

#undef tinyply
#define tinyply _zeno_primplyio_tinyply
#define TINYPLY_IMPLEMENTATION
#include "primplyio_tinyply.h"


/*
static void readply(
    std::vector<zeno::vec3f> &verts,
    std::vector<zeno::vec3f> &color,
    std::vector<zeno::vec3i> &zfaces,
    const std::string & filepath
) {
    std::unique_ptr<std::istream> file_stream;
    std::vector<uint8_t> byte_buffer;

    try {
        file_stream.reset(new std::ifstream(filepath, std::ios::binary));

        if (!file_stream || file_stream->fail()) throw std::runtime_error("file_stream failed to open " + filepath);

        file_stream->seekg(0, std::ios::end);
        const float size_mb = file_stream->tellg() * float(1e-6);
        file_stream->seekg(0, std::ios::beg);

        tinyply::PlyFile file;
        file.parse_header(*file_stream);

        std::shared_ptr<tinyply::PlyData> vertices, r,g,b, faces;

        try { vertices = file.request_properties_from_element("vertex", { "x", "y", "z" }); }
        catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

        try { r = file.request_properties_from_element("vertex", {"red"}); }
        catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

        try { g = file.request_properties_from_element("vertex", {"green"}); }
        catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

        try { b = file.request_properties_from_element("vertex", {"blue"}); }
        catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

        try { faces = file.request_properties_from_element("face", { "vertex_indices" }, 3); }
        catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }


        file.read(*file_stream);

        const size_t numVerticesBytes = vertices->buffer.size_bytes();
        if (vertices->t == tinyply::Type::FLOAT32) {
            verts.resize(vertices->count);
            std::memcpy(verts.data(), vertices->buffer.get(), numVerticesBytes);
        } else if (vertices->t == tinyply::Type::FLOAT64) {
            std::vector<zeno::vec3d> verts_f64;
            verts_f64.resize(vertices->count);
            std::memcpy(verts_f64.data(), vertices->buffer.get(), numVerticesBytes);
            for (const auto& v: verts_f64) {
                verts.emplace_back(v[0], v[1], v[2]);
            }
        }

        if(r->count>0)
        {
          color.resize(verts.size());
          if(r->t == tinyply::Type::UINT8 || r->t ==  tinyply::Type::INT8)
          {
            std::vector<uint8_t> rr;
            std::vector<uint8_t> gg;
            std::vector<uint8_t> bb;
            rr.resize(r->count);
            gg.resize(g->count);
            bb.resize(b->count);
            std::memcpy(rr.data(), r->buffer.get(), r->count );
            std::memcpy(gg.data(), g->buffer.get(), g->count );
            std::memcpy(bb.data(), b->buffer.get(), b->count );

            for(size_t i=0;i<rr.size();i++)
            {
              color[i] = zeno::vec3f(rr[i],gg[i],bb[i])/255.0f;
            }
          }
        }


        const size_t numFacesBytes = faces->buffer.size_bytes();
        if (faces->t == tinyply::Type::INT32 || faces->t == tinyply::Type::UINT32) {
            zfaces.resize(faces->count);
            std::memcpy(zfaces.data(), faces->buffer.get(), numFacesBytes);
        } else if (faces->t == tinyply::Type::INT16 || faces->t == tinyply::Type::UINT16) {
            std::vector<zeno::vec3H> zfaces_uint16;
            zfaces_uint16.res(ce_back(f[0], f[1], f[2]);
            }
        } else if (faces->t == tinyply::Type::INT8 || faces->t == tinyply::Type::UINT8) {
            std::vector<zeno::vec3C> zfaces_uint8;
            zfaces_uint8.resize(faces->count);
            std::memcpy(zfaces_uint8.data(), faces->buffer.get(), numFacesBytes);
            for (const auto& f: zfaces_uint8) {
                zfaces.emplace_back(f[0], f[1], f[2]);
            }
        }

    } catch (const std::exception & e) {
        std::cerr << "Caught tinyply exception: " << e.what() << std::endl;
    }
}
*/
static void ReadAllAttrFromPlyFile(std::string &ply_file, std::shared_ptr<zeno::PrimitiveObject> prim){
    std::filesystem::path file_path(ply_file);
    if(!std::filesystem::exists(file_path)){
        throw std::runtime_error(ply_file + " not exsit");
        return;
    }
    std::ifstream file_stream;
    file_stream.open(file_path);
    if(!file_stream.is_open()){
        throw std::runtime_error("fail to open "+ply_file);
        return;
    }
    tinyply::PlyFile ply_obj;
    if(!ply_obj.parse_header(file_stream)){
        throw std::runtime_error("fail to parse ply header");
        return;
    }
    std::vector<tinyply::PlyElement> elements = ply_obj.get_elements();
    bool found_vertex = false;
    tinyply::PlyElement *vertex_element = nullptr;

    std::vector<tinyply::PlyProperty> need_properties;
    std::vector<std::shared_ptr<tinyply::PlyData>> data_list; 

    int element_size = 0;

    for(tinyply::PlyElement element : elements){
        if(element.name == "vertex"){
            found_vertex = true;
            vertex_element = &element;
            element_size = element.size;
            std::cout << "Name: " <<element.name << std::endl;
            for(tinyply::PlyProperty property : element.properties){
                std::cout << "\tProperty Name: " << property.name ;
                if(property.isList){
                    std::cout << "\tList Type: " << tinyply::PropertyTable[property.listType].str;
                    std::cout << "\tList Size: " << property.listCount;
                }else{
                    need_properties.push_back(property);
                    data_list.push_back(ply_obj.request_properties_from_element("vertex", {property.name}));
                    std::cout << "\t" << tinyply::PropertyTable[property.propertyType].str << "\n";
                }
            }
        std::cout << std::endl;
        }
    }
    if(!found_vertex){
        throw std::runtime_error("No vertex element found in this ply");
        return;
    }

    ply_obj.read(file_stream);
    prim->verts.resize(element_size);

    for(int i=0;i<need_properties.size();i++){
        tinyply::PlyProperty property = need_properties[i];
        auto &new_property = prim->add_attr<float>(property.name);
        unsigned char *buffer = data_list[i]->buffer.get();

        float value = 0.0f;
        tinyply::PropertyInfo info = tinyply::PropertyTable[property.propertyType];

        for(int j=0; j<element_size;j++){
            if(info.str == "char"){
                value = ((char *)buffer)[j];
            }else if(info.str == "uchar"){
                value = ((unsigned char *)buffer)[j];
            }else if(info.str == "short"){
                value = ((short *)buffer)[j];
            }else if(info.str == "ushort"){
                value = ((unsigned short *)buffer)[j];
            }else if(info.str == "int"){
                value = ((int *)buffer)[j];
            }else if(info.str == "uint"){
                value = ((unsigned int *)buffer)[j];
            }else if(info.str == "float"){
                value = ((float *)buffer)[j];
            }else if(info.str == "double"){
                value = ((double *)buffer)[j];
            }else{
                std::cout << "Unknow Type" << std::endl;
            }
            new_property[j] = value;
        }
    }

}

struct ReadPlyPrimitive : zeno::INode {
    virtual void apply() override {
        auto path = get_input<zeno::StringObject>("path")->get();
        auto prim = std::make_shared<zeno::PrimitiveObject>();
        ReadAllAttrFromPlyFile(path, prim);
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(
    ReadPlyPrimitive,
    {
        // inputs
        {
            {
                "readpath",
                "path",
            },
        },
        // outpus
        {
            "prim",
        },
        // params
        {
        },
        // category
        {
            "primitive",
        }
    }
);


static void writeply(
    std::vector<zeno::vec3f> &verts,
    std::vector<zeno::vec3i> &zfaces,
    const std::string & filename
) {
    std::filebuf fb;
    fb.open(filename + ".ply", std::ios::out);
    std::ostream outstream(&fb);
    if (outstream.fail()) {
        throw std::runtime_error("failed to open " + filename);
    }

    tinyply::PlyFile cube_file;

    cube_file.add_properties_to_element("vertex", { "x", "y", "z" }, 
        tinyply::Type::FLOAT32, verts.size(), reinterpret_cast<uint8_t*>(verts.data()), tinyply::Type::INVALID, 0);

    cube_file.add_properties_to_element("face", { "vertex_indices" },
        tinyply::Type::UINT32, zfaces.size(), reinterpret_cast<uint8_t*>(zfaces.data()), tinyply::Type::UINT8, 3);

    cube_file.write(outstream, false);
    
}

struct WritePlyPrimitive : zeno::INode {
    virtual void apply() override {
        std::cerr << "write\n";
        auto path = get_input<zeno::StringObject>("path")->get();
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto &pos = prim->attr<zeno::vec3f>("pos");
        writeply(pos, prim->tris, path.c_str());
    }
};

ZENDEFNODE(
    WritePlyPrimitive,
    {
        // inputs
        {
            {
                "writepath",
                "path",
            },
            "prim",
        },
        // outpus
        {
        },
        // params
        {
        },
        // category
        {
            "primitive",
        }
    }
);
