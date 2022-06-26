#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/DictObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/PrimitiveTools.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/string.h>
#include <zeno/utils/logger.h>
#include <zeno/utils/vec.h>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
//#include <spdlog/spdlog.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#define ROTATE_COMPUTE                          \
    auto gp = glm::vec3(p[0], p[1], p[2]);      \
    gp = mz * my * mx * gp;                     \
    p = zeno::vec3f(gp.x, gp.y, gp.z);

#define ROTATE_PARM                             \
    {"vec3f", "rotate", "0, 0, 0"},

#define ROTATE_MATRIX                           \
    auto rotate = get_input2<zeno::vec3f>("rotate"); \
    float ax = rotate[0] * (M_PI / 180.0);      \
    float ay = rotate[1] * (M_PI / 180.0);      \
    float az = rotate[2] * (M_PI / 180.0);      \
    glm::mat3 mx = glm::mat3(                   \
        1, 0, 0,                                \
        0, cos(ax), -sin(ax),                   \
        0, sin(ax), cos(ax));                   \
    glm::mat3 my = glm::mat3(                   \
        cos(ay), 0, sin(ay),                    \
        0, 1, 0,                                \
        -sin(ay), 0, cos(ay));                  \
    glm::mat3 mz = glm::mat3(                   \
        cos(az), -sin(az), 0,                   \
        sin(az), cos(az), 0,                    \
        0, 0, 1);

namespace zeno {
namespace {
namespace cc4{
    static void flipPrimFaceOrder(PrimitiveObject *prim) {
        for (auto &ind: prim->tris) {
            std::swap(ind[1], ind[2]);
        }
    }
    static std::vector<zeno::vec3f> genindi(int div1, int div2, int inc){
        std::vector<zeno::vec3f> ind;

        for (int i = 0; i < div1-1; i++)
        {
            int i1, i2, i3, i4;
            i1 = i+inc;
            i2 = i1+1;
            i3 = i1+div1;
            i4 = i3+1;
            ind.emplace_back(i1, i3, i2);
            ind.emplace_back(i2, i3, i4);
            for (int j = 0; j < div2-2; j++)
            {
                i1 = div1*(j+1)+i+inc;
                i2 = i1+1;
                i3 = i1+div1;
                i4 = i3+1;
                ind.emplace_back(i1, i3, i2);
                ind.emplace_back(i2, i3, i4);
            }
        }

        return ind;
    }

    static std::vector<zeno::vec3f> igenindi(std::vector<zeno::vec3f>& in, int inc)
    {
        std::vector<zeno::vec3f> out;
        for (int i = 0; i < in.size(); i++)
        {
            out.push_back(in[i]+inc);
        }

        return out;
    }

    static void appind(std::vector<zeno::vec3f> in, zeno::AttrVector<zeno::vec3i>& out){
        for (int i = 0; i < in.size(); i++)
        {
            out.push_back(in[i]);
        }
    }
}
}
}

namespace zeno {
namespace {

struct CreateCube : zeno::INode {
    virtual void apply() override {
        auto prim = std::make_shared<zeno::PrimitiveObject>();
        auto size = get_input2<float>("size");
        auto div_w = get_input2<int>("div_w");
        auto div_h = get_input2<int>("div_h");
        auto div_d = get_input2<int>("div_d");
        auto position = get_input2<zeno::vec3f>("position");
        auto scale = get_input2<zeno::vec3f>("scaleSize");
        ROTATE_MATRIX

        auto &pos = prim->verts;
        auto &tris = prim->tris;
        auto &uv = prim->verts.add_attr<zeno::vec3f>("uv");
        auto &norm = prim->verts.add_attr<zeno::vec3f>("nrm");

        if(div_w <= 2)
            div_w = 2;
        if(div_h <= 2)
            div_h = 2;
        if(div_d <= 2)
            div_d = 2;

        float sw = 1.0 / (div_w-1);
        float sh = 1.0 / (div_h-1);
        float sd = 1.0 / (div_d-1);

        std::vector<zeno::vec3f> fverts;
        std::vector<zeno::vec3f> findis;
        std::vector<zeno::vec3f> bverts;
        std::vector<zeno::vec3f> bindis;
        std::vector<zeno::vec3f> lverts;
        std::vector<zeno::vec3f> lindis;
        std::vector<zeno::vec3f> rverts;
        std::vector<zeno::vec3f> rindis;
        std::vector<zeno::vec3f> uverts;
        std::vector<zeno::vec3f> uindis;
        std::vector<zeno::vec3f> dverts;
        std::vector<zeno::vec3f> dindis;

        std::vector<zeno::vec3f> verts;
        std::vector<zeno::vec3f> indics;
        std::vector<zeno::vec3f> uvs;
        std::vector<zeno::vec3f> normal;


        for (int i = 0; i < div_w; i++)
        {
            for (int j = 0; j < div_h; j++)
            {
                auto p = zeno::vec3f(0.5-i*sw, 0.5-j*sh, -0.5);
                fverts.push_back(p);
                verts.push_back(p);
                uvs.emplace_back(0.375+i*sw*0.25, 0.75+j*sh*0.25, 0);
                normal.emplace_back(0,0,-1);
            }
        }
        for (int i = 0; i < fverts.size(); i++)
        {
            auto fv = fverts[i];
            auto p = zeno::vec3f(fv[0], -fv[1], 0.5);
            bverts.push_back(p);
            verts.push_back(p);
            uvs.emplace_back(uvs[i][0],uvs[i][1]-0.5, 0);
            normal.emplace_back(0,0,1);
        }
        for (int i = 0; i < div_w; i++)
        {
            for (int j = 0; j < div_d; j++)
            {
                auto p = zeno::vec3f(0.5-i*sw, 0.5, 0.5-j*sd);
                uverts.push_back(p);
                verts.push_back(p);
                uvs.emplace_back(0.375+i*sw*0.25, 0.5+j*sd*0.25, 0);
                normal.emplace_back(0,1,0);
            }
        }
        int ui1 = fverts.size()*2;
        for (int i = 0; i < uverts.size(); i++)
        {
            auto uv = uverts[i];
            auto p = zeno::vec3f(uv[0], -uv[1], -uv[2]);
            dverts.push_back(p);
            verts.push_back(p);
            uvs.emplace_back(uvs[i+ui1][0],uvs[i+ui1][1]-0.5, 0);
            normal.emplace_back(0,-1,0);
        }
        for (int i = 0; i < div_h; i++)
        {
            for (int j = 0; j < div_d; j++)
            {
                auto p = zeno::vec3f(0.5, -0.5+i*sh, 0.5-j*sd);
                lverts.push_back(p);
                verts.push_back(p);
                uvs.emplace_back(0.125+i*sh*0.25, j*sd*0.25, 0);
                normal.emplace_back(1,0,0);
            }
        }
        int ui2 = fverts.size()*2+uverts.size()*2;
        for (int i = 0; i < lverts.size(); i++)
        {
            auto lv = lverts[i];
            auto p = zeno::vec3f(-lv[0], -lv[1], lv[2]);
            rverts.push_back(p);
            verts.push_back(p);
            uvs.emplace_back(0.5+uvs[i+ui2][0],uvs[i+ui2][1], 0);
            normal.emplace_back(-1,0,0);
        }

        findis = cc4::genindi(div_h, div_w, 0);
        bindis = cc4::igenindi(findis, fverts.size());
        uindis = cc4::genindi(div_d, div_w,fverts.size()*2);
        dindis = cc4::igenindi(uindis, uverts.size());
        lindis = cc4::genindi(div_d, div_h,fverts.size()*2+uverts.size()*2);
        rindis = cc4::igenindi(lindis, lverts.size());

        cc4::appind(findis, tris);
        cc4::appind(bindis, tris);
        cc4::appind(uindis, tris);
        cc4::appind(dindis, tris);
        cc4::appind(lindis, tris);
        cc4::appind(rindis, tris);

        for (int i = 0; i < verts.size(); i++)
        {
            auto p = verts[i];
            auto n = normal[i];
            auto gn = glm::vec3(n[0], n[1], n[2]);
            p = p * scale * size;
            ROTATE_COMPUTE
            gn = mz * my * mx * gn;
            p = p + position;

            norm.push_back(zeno::vec3f(gn.x, gn.y, gn.z));
            pos.push_back(p);
            uv.push_back(uvs[i]);
        }

        cc4::flipPrimFaceOrder(prim.get());
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(CreateCube, {
    {
        {"vec3f", "position", "0, 0, 0"},
        {"vec3f", "scaleSize", "1, 1, 1"},
        ROTATE_PARM
        {"int", "div_w", "2"},
        {"int", "div_h", "2"},
        {"int", "div_d", "2"},
        {"float", "size", "1"},
    },
    {"prim"},
    {},
    {"create"},
});

struct CreateDisk : zeno::INode {
    virtual void apply() override {
        auto prim = std::make_shared<zeno::PrimitiveObject>();
        auto position = get_input2<zeno::vec3f>("position");
        auto scaleSize = get_input2<zeno::vec3f>("scaleSize");
        auto radius = get_input2<float>("radius");
        auto divisions = get_input2<int>("divisions");

        ROTATE_MATRIX

        auto &verts = prim->verts;
        auto &tris = prim->tris;
        auto &uv = prim->verts.add_attr<zeno::vec3f>("uv");
        auto &norm = prim->verts.add_attr<zeno::vec3f>("nrm");

        if(divisions <= 3){
            divisions = 3;
        }

        verts.emplace_back(zeno::vec3f(0, 0, 0)+position);
        uv.emplace_back(0.5, 0.5, 0);
        norm.emplace_back(0, 1, 0);

        for (int i = 0; i < divisions; i++) {
            float rad = 2 * M_PI * i / divisions;
            auto p = zeno::vec3f(cos(rad) * radius, 0,
                           -sin(rad) * radius);

            ROTATE_COMPUTE

            auto p4uv = p * scaleSize;
            p = p4uv + position;

            verts.emplace_back(p);
            tris.emplace_back(i+1, 0, i+2);
            uv.emplace_back(p4uv[0]/2.0+0.5,
                            p4uv[2]/2.0+0.5, 0);
            norm.emplace_back(0, 1, 0);
        }

        // Update last
        tris[tris.size()-1] = zeno::vec3i(divisions, 0, 1);

        cc4::flipPrimFaceOrder(prim.get());
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(CreateDisk, {
    {
        {"vec3f", "position", "0, 0, 0"},
        {"vec3f", "scaleSize", "1, 1, 1"},
        ROTATE_PARM
        {"float", "radius", "1"},
        {"int", "divisions", "32"},
    },
    {"prim"},
    {},
    {"create"},
});

struct CreatePlane : zeno::INode {
    virtual void apply() override {
        auto prim = std::make_shared<zeno::PrimitiveObject>();
        auto position = get_input2<zeno::vec3f>("position");
        auto scale = get_input2<zeno::vec3f>("scaleSize");
        auto size = get_input2<float>("size");
        auto rows = get_input2<int>("rows");;
        auto columns = get_input2<int>("columns");;

        ROTATE_MATRIX

        auto &verts = prim->verts;
        auto &tris = prim->tris;
        std::vector<zeno::vec3f> uvs;
        std::vector<zeno::vec3f> nors;

        if(rows <= 1)
            rows = 1;
        if(columns <= 1)
            columns = 1;

        auto start_point = zeno::vec3f(0.5, 0, 0.5);
        auto gscale = glm::vec3(scale[0], scale[1], scale[2]);
        auto gposition = glm::vec3(position[0], position[1], position[2]);
        zeno::vec3f normal(0.0f);
        float rm = 1.0 / rows;
        float cm = 1.0 / columns;
        int fi = 0;

        // Vertices & UV
        for(int i=0; i<=rows; i++){

            auto rp = start_point - zeno::vec3f(i*rm, 0, 0);

            for(int j=0; j<=columns; j++){
                auto p = rp - zeno::vec3f(0, 0, j*cm);

                ROTATE_COMPUTE

                auto zcp = zeno::vec3f(p[0], p[1], p[2]);
                zcp = zcp * scale + position;
                zcp = zcp * size;
                verts.push_back(zcp);
                uvs.emplace_back(i*rm, j*cm*-1+1, 0);
            }
        }

        // Indices
        for(int i=0; i<rows; i++){
            for(int j=0; j<columns; j++){
                int i1 = fi;
                int i2 = i1+1;
                int i3 = fi+(columns+1);
                int i4 = i3+1;

                tris.emplace_back(i1, i3, i2);
                tris.emplace_back(i2, i3, i4);

                fi += 1;
            }
            fi += 1;
        }

        // Normal
        for(int i=0; i<1; i++){
            auto ind = tris[i];
            // 0,3,1
            auto pos1 = verts[int(ind[0])];
            auto pos2 = verts[int(ind[1])];
            auto pos3 = verts[int(ind[2])];

            auto uv1 = uvs[int(ind[0])];
            auto uv2 = uvs[int(ind[1])];
            auto uv3 = uvs[int(ind[2])];

            auto edge1 = pos2 - pos1;
            auto edge2 = pos3 - pos1;
            auto deltaUV1 = uv2 - uv1;
            auto deltaUV2 = uv3 - uv1;

            zeno::vec3f tangent1, bitangent1;

            float f = 1.0f / (deltaUV1[0] * deltaUV2[1] - deltaUV2[0] * deltaUV1[1]);

            tangent1[0] = f * (deltaUV2[1] * edge1[0] - deltaUV1[1] * edge2[0]);
            tangent1[1] = f * (deltaUV2[1] * edge1[1] - deltaUV1[1] * edge2[1]);
            tangent1[2] = f * (deltaUV2[1] * edge1[2] - deltaUV1[1] * edge2[2]);

            bitangent1[0] = f * (-deltaUV2[0] * edge1[0] + deltaUV1[0] * edge2[0]);
            bitangent1[1] = f * (-deltaUV2[0] * edge1[1] + deltaUV1[0] * edge2[1]);
            bitangent1[2] = f * (-deltaUV2[0] * edge1[2] + deltaUV1[0] * edge2[2]);

            normal = cross(tangent1, bitangent1);
            //normal = normalize(cross(edge2, edge1));
        }

        // Assign uv & normal
        auto &uv = prim->verts.add_attr<zeno::vec3f>("uv");
        auto &norm = prim->verts.add_attr<zeno::vec3f>("nrm");
        for(int i=0; i<verts.size(); i++){
            uv[i] = uvs[i];
            norm[i] = normal;
        }

        cc4::flipPrimFaceOrder(prim.get());
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(CreatePlane, {
    {
        {"vec3f", "position", "0, 0, 0"},
        {"vec3f", "scaleSize", "1, 1, 1"},
        ROTATE_PARM
        {"float", "size", "1"},
        {"int", "rows", "2"},
        {"int", "columns", "2"},
    },
    {"prim"},
    {},
    {"create"},
});

struct CreateTube : zeno::INode {
    virtual void apply() override {
        auto prim = std::make_shared<zeno::PrimitiveObject>();

        auto position = get_input2<zeno::vec3f>("position");
        auto scale = get_input2<zeno::vec3f>("scaleSize");
        auto radius1 = get_input2<float>("radius1");
        auto radius2 = get_input2<float>("radius2");
        auto height = get_input2<float>("height");
        auto rows = get_input2<int>("rows");
        auto columns = get_input2<int>("columns");
        ROTATE_MATRIX

        auto &verts = prim->verts;
        auto &indis = prim->tris;
        auto &uvs = prim->verts.add_attr<zeno::vec3f>("uv");
        auto &normal = prim->verts.add_attr<zeno::vec3f>("nrm");

        if(rows <= 2)
            rows = 2;
        if(columns <= 3)
            columns = 3;

        std::vector<zeno::vec3f> mverts;
        std::vector<int> svi;
        std::vector<int> cvi;

        float hs = height/2.0;
        float hm = height / 1.0/(rows-1);

        verts.emplace_back(0,hs,0);
        normal.emplace_back(0,1,0);
        uvs.emplace_back(0.5, 0.85, 0);
        for (int i = 0; i < columns; i++) {
            float rad = 2 * M_PI * i / columns;
            float x = cos(rad);
            float z = -sin(rad);
            float of = 0.125;
            auto p1 = zeno::vec3f(x*radius1, hs, z*radius1);
            auto p2 = zeno::vec3f(x*radius2, -hs, z*radius2);
            auto n1 = zeno::vec3f(0,1,0);
            auto n2 = zeno::vec3f(0,-1,0);
            verts.push_back(p1);
            verts.push_back(p2);
            normal.push_back(n1);
            normal.push_back(n2);
            uvs.emplace_back(
                x*0.5*3/10+0.5,
                z*0.5*3/10+0.85, 0);
            uvs.emplace_back(
                x*0.5*3/10+0.5,
                z*0.5*3/10+0.15, 0);
        }
        verts.emplace_back(0,-hs,0);
        normal.emplace_back(0,-1,0);
        uvs.emplace_back(0.5, 0.15, 0);

        for (int i = 1; i < columns+1; i++) {
            int i1, i2, i3;
            i1 = i*2-1;
            i2 = 0;
            i3 = i1+2;
            if(i == columns)
                i3 = 1;
            indis.emplace_back(i1, i2, i3);

            i1 = i*2;
            i2 = verts.size()-1;
            i3 = i1+2;
            if(i == columns)
                i3 = 2;
            indis.emplace_back(i3, i2, i1);
        }

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                float f = float(i)/(rows-1);
                int it = 1+2*j;
                int id = 2+2*j;
                auto pu = verts[it];
                auto pd = verts[id];
                auto np = (1-f)*pu+(f)*pd;

                if(j == 0){
                    mverts.emplace_back(np[0], np[1], np[2]);
                    normal.emplace_back(-1,0,0);
                    int fi = verts.size()+mverts.size()-1;
                    svi.push_back(fi);
                    cvi.push_back(fi+1);
                }

                mverts.emplace_back(np[0], np[1], np[2]);
                normal.emplace_back(-1,0,0);
            }
        }

        for (int i = 0; i < rows+1; i++)
        {
            for (int j = 0; j < columns+1; j++)
            {
                float rf = 1.0/(rows-1);
                float cf = 1.0/(columns);
                float sf = 4.0/10;
                float uf = 0.3;
                auto uv = zeno::vec3f(
                    (j-1)*cf*sf+uf,
                    i*rf*sf+uf, 0);

                if(j == 0){
                    uvs.emplace_back(1*sf+uf, uv[1], 0);
                }else if(j == 1){
                    uvs.emplace_back(0*sf+uf, uv[1], 0);
                }
                else{
                    uvs.emplace_back(uv);
                }
            }
        }

        for (int i = 0; i < mverts.size(); i++)
        {
            verts.push_back(mverts[i]);
        }

        for (int i = 0; i < rows-1; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                int i1, i2, i3, i4;
                i1 = j+i*(columns+1)+1;
                i2 = i1+columns+1;
                i3 = i1+1;
                i4 = i2+1;
                if(j == columns-1){
                    i3 = i3 - columns-1;
                    i4 = i4 - columns-1;
                }
                i1 += (verts.size() - mverts.size());
                i2 += (verts.size() - mverts.size());
                i3 += (verts.size() - mverts.size());
                i4 += (verts.size() - mverts.size());

                int fi = i1-1;
                if(j == 0){
                    fi += columns;
                }
                auto p1 = verts[i1]*scale;
                auto p2 = verts[i2]*scale;
                auto p3 = verts[i3]*scale;
                auto pn = verts[fi]*scale;

                auto n1 = normalize(cross(p2-p1, p3-p1));
                auto n2 = normalize(cross(pn-p1, p2-p1));

                n1 = normalize((n1+n2)/2.0);

                normal[i1] = n1;
                normal[i2] = n1;
                normal[i3] = n1;
                normal[i4] = n1;

                indis.emplace_back(i1, i3, i2);
                indis.emplace_back(i4, i2, i3);
            }
        }

        for (int i = 0; i < svi.size(); i++)
        {
            normal[svi[i]] = normal[cvi[i]];
        }

        for (int i = 0; i < verts->size(); i++)
        {
            auto p = verts[i];
            auto n = normal[i];

            auto gp = glm::vec3(p[0], p[1], p[2]);
            gp = mz * my * mx * gp;
            p = zeno::vec3f(gp.x, gp.y, gp.z);

            auto gn = glm::vec3(n[0], n[1], n[2]);
            gn = mz * my * mx * gn;
            n = zeno::vec3f(gn.x, gn.y, gn.z);

            verts[i] = p * scale + position;
            normal[i] = n;
        }

        cc4::flipPrimFaceOrder(prim.get());
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(CreateTube, {
    {
        {"vec3f", "position", "0, 0, 0"},
        {"vec3f", "scaleSize", "1, 1, 1"},
        ROTATE_PARM
        {"float", "radius1", "1"},
        {"float", "radius2", "1"},
        {"float", "height", "2"},
        {"int", "rows", "2"},
        {"int", "columns", "12"}
    },
    {"prim"},
    {},
    {"create"},
});

struct CreateSphere : zeno::INode {
    virtual void apply() override {
        auto prim = std::make_shared<zeno::PrimitiveObject>();
        auto position = get_input2<zeno::vec3f>("position");
        auto scale = get_input2<zeno::vec3f>("scaleSize");
        auto rows = get_input2<int>("rows");
        auto columns = get_input2<int>("columns");
        auto radius = get_input2<float>("radius");

        ROTATE_MATRIX

        if(rows <= 3)
            rows = 3;
        if(columns <= 3)
            columns = 3;

        //auto &uv = prim->add_attr<zeno::vec3f>("uv");
        //auto &nrm = prim->add_attr<zeno::vec3f>("nrm");
        auto &uv = prim->verts.add_attr<zeno::vec3f>("uv");
        auto &nrm = prim->verts.add_attr<zeno::vec3f>("nrm");

        int c = 0;
        int tp = rows * columns;
        float row_sep = 180.0f / (rows - 1);

        for (int i = 0; i<rows; i++) {
            float ic = -90.0f + i*row_sep;
            float r = std::cos(ic / 180.0 * M_PI);
            float h = std::sin(ic / 180.0 * M_PI);

            for (int j = 0; j < columns; j++) {
                float rad = 2 * M_PI * j / columns;
                // position
                zeno::vec3f op = zeno::vec3f (
                    cos(rad) * r,
                    h,
                    sin(rad) * r);
                zeno::vec3f p = op * scale * radius;

                ROTATE_COMPUTE

                p = p + position;
                zeno::vec3f np = p * scale * radius;
                prim->verts.push_back(p);

                // normal
                zeno::vec3f n;
                n = zeno::normalize(np - zeno::vec3f(0,0,0));
                nrm.push_back(n);

                // uv
                zeno::vec3f uvw;
                float u,v;
                if(i == 0){
                    u = float(j)/(columns-1);
                    v = 0.0;

                }else if(i == rows-1){
                    u = float(j)/(columns-1);
                    v = 1.0;
                }else{
                    u = -1.0;
                    v = -1.0;
                }
                uv.emplace_back(u,v,0);
                if(j == 0 && i > 0 && i < rows-1){
                    prim->verts.push_back(p);
                    nrm.push_back(n);
                    uv.emplace_back(u,v,0);
                    c+=1;
                }

                // indices
                if(i == 0){
                    int i1 = c;//bottom
                    int i2 = c+columns+2;
                    int i3 = c+columns+1;
                    if(i2 >= 2*columns+1)
                        i2 = columns;
                    int i4 = tp-columns+j+rows-2;//top
                    int _t = tp-2*columns+rows-2;
                    int _t1 = tp-columns+rows-2;
                    int i5 = _t+j;
                    int i6 = _t+j+1;
                    if(i6>=_t1)
                        i6 -= columns+1;

                    prim->tris.push_back(zeno::vec3i(i1, i2, i3));
                    prim->tris.push_back(zeno::vec3i(i4, i5, i6));
                }

                if(rows > 3 && i < rows-2 && i>0){

                    int i1 = c;
                    int i2 = c+1;
                    int i3 = c+columns+1;

                    int i4 = i3;
                    int i5 = i1+columns+2;

                    if(j == columns-1){
                        i2 -= columns+1;
                        i5 -= columns+1;
                    }
                    int i6 = i2;

                    prim->tris.push_back(zeno::vec3i(i1, i2, i3));
                    prim->tris.push_back(zeno::vec3i(i4, i6, i5));
                }
                c+=1;
            }
        }

        float vi = 0;
        float s = 1.0f/(rows-1);
        for(int i=columns; i<(prim->verts.size() - columns); i++){

            int id = (i-columns)%(columns+1);
            if(id == 0)
                vi += 1;
            float u,v;
            if(id-1 < 0)
                u = 1.0;
            else
                u = float(id-1)/(columns);
            v = s*vi;
            uv[i] = zeno::vec3f(u,v,0);
        }

        cc4::flipPrimFaceOrder(prim.get());
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(CreateSphere, {
    {
        {"vec3f", "position", "0, 0, 0"},
        {"vec3f", "scaleSize", "1, 1, 1"},
        {"float", "radius", "1"},
        ROTATE_PARM
        {"int", "rows", "13"},
        {"int", "columns", "24"},
    },
    {"prim"},
    {},
    {"create"},
});

struct CreateCone : zeno::INode {
    virtual void apply() override {
        auto prim = std::make_shared<zeno::PrimitiveObject>();
        auto position = get_input2<zeno::vec3f>("position");
        auto scaleSize = get_input2<zeno::vec3f>("scaleSize");
        auto radius = get_input2<float>("radius");
        auto height = get_input2<float>("height");
        auto lons = get_input2<int>("lons");

        auto &pos = prim->verts;
        for (size_t i = 0; i < lons; i++) {
            float rad = 2 * M_PI * i / lons;
            pos.push_back(vec3f(cos(rad) * radius, -0.5 * height, -sin(rad) * radius) * scaleSize + position);
        }
        // top
        pos.push_back(vec3f(0, 0.5 * height, 0) * scaleSize + position);
        // bottom
        pos.push_back(vec3f(0, -0.5 * height, 0) * scaleSize + position);

        auto &tris = prim->tris;
        for (size_t i = 0; i < lons; i++) {
            tris.push_back(vec3i(lons, i, (i + 1) % lons));
            tris.push_back(vec3i(i, lons + 1, (i + 1) % lons));
        }

        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(CreateCone, {
    {
        {"vec3f", "position", "0, 0, 0"},
        {"vec3f", "scaleSize", "1, 1, 1"},
        {"float", "radius", "1"},
        {"float", "height", "2"},
        {"int", "lons", "32"},
    },
    {"prim"},
    {},
    {"create"},
});

struct CreateCylinder : zeno::INode {
    virtual void apply() override {
        auto prim = std::make_shared<zeno::PrimitiveObject>();

        auto position = get_input2<zeno::vec3f>("position");
        auto scaleSize = get_input2<zeno::vec3f>("scaleSize");
        auto radius = get_input2<float>("radius");
        auto height = get_input2<float>("height");
        auto lons = get_input2<int>("lons");

        auto &pos = prim->verts;
        for (size_t i = 0; i < lons; i++) {
            float rad = 2 * M_PI * i / lons;
            pos.push_back(vec3f(cos(rad) * radius, 0.5 * height, -sin(rad) * radius) * scaleSize + position);
        }
        for (size_t i = 0; i < lons; i++) {
            float rad = 2 * M_PI * i / lons;
            pos.push_back(vec3f(cos(rad) * radius, -0.5 * height, -sin(rad) * radius) * scaleSize + position);
        }
        pos.push_back(vec3f(0, 0.5 * height, 0) * scaleSize + position);
        pos.push_back(vec3f(0, -0.5 * height, 0) * scaleSize + position);

        auto &tris = prim->tris;
        // Top
        for (size_t i = 0; i < lons; i++) {
            tris.push_back(vec3i(lons * 2, i, (i + 1) % lons));
        }
        // Bottom
        for (size_t i = 0; i < lons; i++) {
            tris.push_back(vec3i(i + lons, lons * 2 + 1, (i + 1) % lons + lons));
        }
        // Side
        for (size_t i = 0; i < lons; i++) {
            size_t _0 = i;
            size_t _1 = (i + 1) % lons;
            size_t _2 = (i + 1) % lons + lons;
            size_t _3 = i + lons;
            tris.push_back(vec3i(_1, _0, _2));
            tris.push_back(vec3i(_2, _0, _3));
        }
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(CreateCylinder, {
    {
        {"vec3f", "position", "0, 0, 0"},
        {"vec3f", "scaleSize", "1, 1, 1"},
        {"float", "radius", "1"},
        {"float", "height", "2"},
        {"int", "lons", "32"},
    },
    {"prim"},
    {},
    {"create"},
});

}
}
