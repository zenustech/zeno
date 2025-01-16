#include "glm/matrix.hpp"
#include "glm/gtx/string_cast.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
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
#include <zeno/utils/log.h>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <glm/glm.hpp>

#define SH_C0 0.28209479177387814f
#undef tinyply
#define tinyply _zeno_primplyio_tinyply
#define TINYPLY_IMPLEMENTATION
#include "primplyio_tinyply.h"

#include "happly.h"

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
glm::mat3 getTransform(glm::vec3 scale, glm::vec4 q, bool print=false)  // should be correct
{
    glm::mat3 S = glm::mat3(0.f);
    S[0][0] = scale.x;
	S[1][1] = scale.y;
	S[2][2] = scale.z;
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

    glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);
    glm::mat3 M =  S * R ;
    glm::mat3 Sigma = glm::transpose(R) *  S * R;

    if(print){
        std::cout << "R = " << glm::to_string(R) <<std::endl;
        std::cout << "S = " << glm::to_string(S) <<std::endl;
        std::cout << "M = " << glm::to_string(M) <<std::endl;
        std::cout << "Sigma = " << glm::to_string(Sigma) <<std::endl;
    }

    return Sigma;
}

static void ReadGassionSplattingFromPly(std::string &ply_file, std::shared_ptr<zeno::PrimitiveObject> prim, bool preview, float preScale=1.0f){
    std::filesystem::path file_path(ply_file);
    if(!std::filesystem::exists(file_path)){
        throw std::runtime_error(ply_file + " not exsit");
        return;
    }
    
    std::ifstream file_stream;
    file_stream.open(file_path,std::ios::binary);
    if(!file_stream.is_open()){
        throw std::runtime_error("fail to open "+ply_file);
        return;
    }
    
    happly::PLYData ply_obj(file_stream);
    try{
        ply_obj.validate();
    }catch(std::exception &e){
        std::cerr << e.what() << std::endl;
        throw e;
        return;
    }

    happly::Element& vertex = ply_obj.getElement("vertex");
    std::cout << "Vertex count = " <<vertex.count << std::endl;
    size_t vertex_count = vertex.count;
    std::vector<zeno::vec3f> &color = prim->add_attr<zeno::vec3f>("clr");
    std::vector<float> &opacity = prim->add_attr<float>("opacity");
    std::vector<zeno::vec3f> &scale = prim->add_attr<zeno::vec3f>("scale");
    std::vector<zeno::vec4f> &rotate = prim->add_attr<zeno::vec4f>("rotate");
    std::vector<zeno::vec3f> &nrm = prim->add_attr<zeno::vec3f>("nrm");
    std::vector<zeno::vec3f> &tang = prim->add_attr<zeno::vec3f>("tang");
    std::vector<std::vector<float>*> SH_attrs;
    SH_attrs.resize(48);
    for(int i=0;i<48;i++){
        char c_str[10];
        snprintf(c_str, 10,"SH_%d",i);
        std::cout << c_str << std::endl;
        SH_attrs[i] = & prim->add_attr<float>(c_str);
        SH_attrs[i]->resize(vertex_count);
    }
    prim->verts.resize(vertex_count);
    std::cout << "Vertex count = " <<vertex_count << std::endl;

    try{
        std::vector<float> pos_x = vertex.getProperty<float>("x");
        std::vector<float> pos_y = vertex.getProperty<float>("y");
        std::vector<float> pos_z = vertex.getProperty<float>("z");

        std::vector<float> op = vertex.getProperty<float>("opacity");
        std::vector<float> scale_x = vertex.getProperty<float>("scale_0");
        std::vector<float> scale_y = vertex.getProperty<float>("scale_1");
        std::vector<float> scale_z = vertex.getProperty<float>("scale_2");

        std::vector<float> rot_0 = vertex.getProperty<float>("rot_0");
        std::vector<float> rot_1 = vertex.getProperty<float>("rot_1");
        std::vector<float> rot_2 = vertex.getProperty<float>("rot_2");
        std::vector<float> rot_3 = vertex.getProperty<float>("rot_3");

        std::vector<std::vector<float>> SH_params;
        SH_params.resize(48);
        SH_params[0] = vertex.getProperty<float>("f_dc_0");
        SH_params[1] = vertex.getProperty<float>("f_dc_1");
        SH_params[2] = vertex.getProperty<float>("f_dc_2");
        for(int i=0;i<45;i++){
            char c_str[15] = "";
            snprintf(c_str, 15,"f_rest_%d",i);
            std::string str(c_str);
            SH_params[i+3]= vertex.getProperty<float>(str);
        }

        #pragma omp parallel for
        for(auto i=0;i<vertex_count;i++){
            scale_x[i] = exp(scale_x[i]);
            scale_y[i] = exp(scale_y[i]);
            scale_z[i] = exp(scale_z[i]);

            zeno::vec3f pos(pos_x[i],pos_y[i],pos_z[i]);
            prim->verts[i]= pos * preScale;

            for(int j=0;j<48;j++){
                (*SH_attrs[j])[i]=(SH_params[j][i]);
            }

            zeno::vec3f current_scale = zeno::vec3f(scale_x[i],scale_y[i],scale_z[i]) * preScale;
            zeno::vec4f current_rotate = zeno::vec4f(rot_0[i],rot_1[i],rot_2[i],rot_3[i]);
            current_rotate = zeno::normalizeSafe(current_rotate);

            if(preview){
                float r = std::clamp(0.5f + SH_C0 * SH_params[0][i],0.0f,1.0f);
                float g = std::clamp(0.5f + SH_C0 * SH_params[1][i],0.0f,1.0f);
                float b = std::clamp(0.5f + SH_C0 * SH_params[2][i],0.0f,1.0f);
                color[i] = zeno::vec3f(r,g,b);
            }else{
                //mat = {nrm,clr,tang}
                glm::vec3 scale ={current_scale[0],current_scale[1],current_scale[2]};
                glm::vec4 rotate ={current_rotate[0],current_rotate[1],current_rotate[2],current_rotate[3]};
                glm::mat3 mat;
                if(i==0){
                    zeno::log_info("pos = {}",pos);
                    mat = getTransform(scale, rotate,true);
                    zeno::log_info("tang = {},{},{}",mat[2][0],mat[2][1],mat[2][2]);
                }else{
                    mat = getTransform(scale, rotate);
                }
                nrm[i] = zeno::vec3f(mat[0][0],mat[0][1],mat[0][2]);
                color[i] = zeno::vec3f(mat[1][0],mat[1][1],mat[1][2]);
                tang[i] = zeno::vec3f(mat[2][0],mat[2][1],mat[2][2]);
            }
            opacity[i] = 1.0f/(1 + exp(- op[i]));
            scale[i] = current_scale;
            rotate[i] = current_rotate;
        }

    }catch(std::exception &e){
        std::cerr << e.what() << std::endl;
        throw e;
        return;
    }
    return;
}


static void ReadAllAttrFromPlyFile(std::string &ply_file, std::shared_ptr<zeno::PrimitiveObject> prim){
    std::filesystem::path file_path(ply_file);
    if(!std::filesystem::exists(file_path)){
        throw std::runtime_error(ply_file + " not exsit");
        return;
    }
    
    std::ifstream file_stream;
    file_stream.open(file_path,std::ios::binary);
    if(!file_stream.is_open()){
        throw std::runtime_error("fail to open "+ply_file);
        return;
    }
    
    happly::PLYData ply_obj(file_stream);
    try{
        ply_obj.validate();
    }catch(std::exception &e){
        std::cerr << e.what() << std::endl;
        throw e;
        return;
    }
    std::vector<std::string> element_names = ply_obj.getElementNames();
    for(std::string element_name : element_names){
        std::cout << element_name << "\n  |\n";
        std::vector<std::string>property_names = ply_obj.getElement(element_name).getPropertyNames();
        for(std::string property_name : property_names){
            std::cout << "\t|--" << property_name << std::endl;
        }
        
    }
    happly::Element& vertex = ply_obj.getElement("vertex");
    std::cout << "Vertex count = " <<vertex.count << std::endl;
    prim->verts.resize(vertex.count);
    std::vector<std::string> property_names = vertex.getPropertyNames();
    for(std::string property_name : property_names){
        std::vector<float> &new_property = prim->add_attr<float>(property_name);
        try{
            std::vector<float> data = vertex.getProperty<float>(property_name);
            new_property.assign(data.begin(),data.end());
        }catch(std::exception &e){
            std::cerr << e.what() << std::endl;
            throw e;
            return;
        }
    }
    return;

}

struct ReadGassionSplatting : zeno::INode {
    virtual void apply() override {
        auto path = get_input<zeno::StringObject>("path")->get();
        auto prim = std::make_shared<zeno::PrimitiveObject>();
        bool preview = get_input2<bool>("preview");
        float pScale = get_input2<float>("preScale");
        ReadGassionSplattingFromPly(path, prim,preview,pScale);
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(
    ReadGassionSplatting,
    {
        // inputs
        {
            {
                "readpath",
                "path",
            },
            {"float","preScale","1"},
            {"bool","preview","0"}
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

struct GassionExample : zeno::INode {
    virtual void apply() override {
        auto prim = std::make_shared<zeno::PrimitiveObject>();
        bool preview = get_input2<bool>("preview");
        prim->verts->resize(4);
        prim->verts[0] = zeno::vec3f(0,0,0);
        prim->verts[1] = zeno::vec3f(1,0,0);
        prim->verts[2] = zeno::vec3f(0,1,0);
        prim->verts[3] = zeno::vec3f(0,0,1);
        std::vector<zeno::vec3f> &color = prim->verts.add_attr<zeno::vec3f>("clr");
        std::vector<zeno::vec3f> &nrm = prim->verts.add_attr<zeno::vec3f>("nrm");
        std::vector<zeno::vec3f> &tang = prim->verts.add_attr<zeno::vec3f>("tang");


        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(
    GassionExample,
    {
        // inputs
        {
            {"bool","preview","0"}
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
