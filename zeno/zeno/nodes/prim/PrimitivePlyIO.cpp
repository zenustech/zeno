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

#define TINYPLY_IMPLEMENTATION
#include "tinyply.h"


static void readply(
    std::vector<zeno::vec3f> &verts,
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

        std::shared_ptr<tinyply::PlyData> vertices, faces;

        try { vertices = file.request_properties_from_element("vertex", { "x", "y", "z" }); }
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
        const size_t numFacesBytes = faces->buffer.size_bytes();
        if (faces->t == tinyply::Type::INT32 || faces->t == tinyply::Type::UINT32) {
            zfaces.resize(faces->count);
            std::memcpy(zfaces.data(), faces->buffer.get(), numFacesBytes);
        } else if (faces->t == tinyply::Type::INT16 || faces->t == tinyply::Type::UINT16) {
            std::vector<zeno::vec3H> zfaces_uint16;
            zfaces_uint16.resize(faces->count);
            std::memcpy(zfaces_uint16.data(), faces->buffer.get(), numFacesBytes);
            for (const auto& f: zfaces_uint16) {
                zfaces.emplace_back(f[0], f[1], f[2]);
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

struct ReadPlyPrimitive : zeno::INode {
    virtual void apply() override {
        auto path = get_input<zeno::StringObject>("path")->get();
        auto prim = std::make_shared<zeno::PrimitiveObject>();
        auto &pos = prim->verts;
        auto &tris = prim->tris;
        readply(pos, tris, path);
        prim->resize(pos.size());
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