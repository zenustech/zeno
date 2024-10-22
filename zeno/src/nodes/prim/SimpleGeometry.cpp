#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/DictObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/PrimitiveTools.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/MatrixObject.h>
#include <zeno/types/UserData.h>
#include <zeno/utils/string.h>
#include <zeno/utils/logger.h>
#include <zeno/utils/vec.h>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
//#include <spdlog/spdlog.h>

#include <zeno/utils/eulerangle.h>

#include <glm/glm.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <filesystem>
#include <cstdlib>

#define ROTATE_COMPUTE                          \
    auto gp = glm::vec3(p[0], p[1], p[2]);      \
    gp = mz * my * mx * gp;                     \
    p = zeno::vec3f(gp.x, gp.y, gp.z);

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

#define NORMUV_CIHOU                            \
    if (!get_input2<bool>("isFlipFace"))        \
        cc4::flipPrimFaceOrder(prim.get());     \
    if (!get_input2<bool>("hasNormal"))         \
        prim->verts.attrs.erase("nrm");         \
    if (!get_input2<bool>("hasVertUV"))         \
        prim->verts.attrs.erase("uv");

namespace zeno {
namespace {
namespace cc4{
    static void flipPrimFaceOrder(PrimitiveObject *prim) {
        for (auto &ind: prim->tris) {
            std::swap(ind[1], ind[2]);
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
        auto quad = get_input2<bool>("quads");
        auto position = get_input2<zeno::vec3f>("position");
        auto scale = get_input2<zeno::vec3f>("scaleSize");
        ROTATE_MATRIX

        auto &verts = prim->verts;
        auto &indis = prim->tris;
        auto &quads = prim->quads;
        auto &nors = prim->verts.add_attr<zeno::vec3f>("nrm");
        auto &poly = prim->polys;
        auto &loops = prim->loops;

        std::vector<zeno::vec3f> dummy;
        auto &uv1 = !quad ?  prim->tris.add_attr<zeno::vec3f>("uv0") : dummy;
        auto &uv2 = !quad ?  prim->tris.add_attr<zeno::vec3f>("uv1") : dummy;
        auto &uv3 = !quad ?  prim->tris.add_attr<zeno::vec3f>("uv2") : dummy;

        if(div_w <= 2)
            div_w = 2;
        if(div_h <= 2)
            div_h = 2;
        if(div_d <= 2)
            div_d = 2;

        float sw = 1.0f / (div_w-1);
        float sh = 1.0f / (div_h-1);
        float sd = 1.0f / (div_d-1);
        float sc=0.25f;

        for (int i = 0; i < div_w; i++)
        {
            for (int j = 0; j < div_h; j++)
            {
                for (int k = 0; k < div_d; k++)
                {
                    auto p = zeno::vec3f(
                        0.5f-i*sw, -0.5f+j*sh, 0.5f-k*sd);

                    if(j == 0 || j == div_h-1 || i == 0 || i == div_w-1 || k == 0 || k == div_d-1){
                        verts.push_back(p);
                        nors.emplace_back(0.0f,0.0f,0.0f);
                    }
                }
            }
        }

        int le =div_h*div_d;
        int rls=verts.size()-div_d;
        int pp=(div_h-1)*div_d;
        int pcircle=le-(div_h-2)*(div_d-2);
        int inc=verts.size()-div_h*div_d;

        int ss=0;
        for (int j = 0; j < div_h-1; j++)
        {
            for (int k = 0; k < div_d-1; k++)
            {
                int a =j*div_h+k*div_d;

                int i1,i2,i3,i4;
                i1=k+j*(div_d);
                i2=i1+1;
                i3=i2+(div_d-1);
                i4=i3+1;

                float u1,v1,u2,v2,u3,v3,u4,v4;
                zeno::vec3f uvw1,uvw2,uvw3,uvw4;

                v1=1.0f-k*sd;
                u1=1.0f-sh*j;
                u2=u1;
                v2=v1-sd;
                u3=u1-sh;
                v3=v1;
                u4=u3;
                v4=v2;

                uvw1=zeno::vec3f(0.125+u1*sc,v1*sc,0);
                uvw2=zeno::vec3f(0.125+u2*sc,v2*sc,0);
                uvw3=zeno::vec3f(0.125+u3*sc,v3*sc,0);
                uvw4=zeno::vec3f(0.125+u4*sc,v4*sc,0);

                // Left
                if(quad){
                    prim->loops.push_back(i1);
                    prim->loops.push_back(i2);
                    prim->loops.push_back(i4);
                    prim->loops.push_back(i3);
                    prim->polys.push_back({ss * 4, 4});
                }else{
                    indis.emplace_back(i1,i3,i2);
                    indis.emplace_back(i4,i2,i3);
                    uv1.push_back(uvw1);uv2.push_back(uvw3);uv3.push_back(uvw2);
                    uv1.push_back(uvw4);uv2.push_back(uvw2);uv3.push_back(uvw3);
                }
                ss++;

                uvw1=zeno::vec3f((1.0f-u1)*sc+0.625f,v1*sc,0);
                uvw2=zeno::vec3f((1.0f-u2)*sc+0.625f,v2*sc,0);
                uvw3=zeno::vec3f((1.0f-u3)*sc+0.625f,v3*sc,0);
                uvw4=zeno::vec3f((1.0f-u4)*sc+0.625f,v4*sc,0);

                int i1_=i1+inc,i2_=i2+inc,i3_=i3+inc,i4_=i4+inc;

                if(k!=div_d-2&&j!=div_h-2){
                    nors[i4]=zeno::vec3f(1,0,0);
                    nors[i4_]=zeno::vec3f(-1,0,0);
                }

                // Right
                if(quad){
                    prim->loops.push_back(i3_);
                    prim->loops.push_back(i4_);
                    prim->loops.push_back(i2_);
                    prim->loops.push_back(i1_);
                    prim->polys.push_back({ss * 4, 4});
                }else{
                    indis.emplace_back(i1_,i2_,i3_);
                    indis.emplace_back(i4_,i3_,i2_);
                    uv1.push_back(uvw1);uv2.push_back(uvw2);uv3.push_back(uvw3);
                    uv1.push_back(uvw4);uv2.push_back(uvw3);uv3.push_back(uvw2);
                }
                ss++;
            }
        }

        for (int j = -1; j < div_w-2; j++)
        {
            for (int i = 0; i < div_d-1; i++)
            {
                int i1,i2,i3,i4;
                i1=i+le+j*pcircle;
                i2=i1+pcircle;
                if(j==-1)
                    i1=i;
                i3=i1+1;
                i4=i2+1;

                float u1,v1,u2,v2,u3,v3,u4,v4;
                zeno::vec3f uvw1,uvw2,uvw3,uvw4;

                u1=sw*(j+1);
                v1=1.0f-i*sd;
                u2=u1+sw;
                v2=v1;
                u3=u1;
                v3=v1-sd;
                u4=u2;
                v4=v3;

                uvw1=zeno::vec3f(0.375f+u1*sc,v1*sc,0);
                uvw2=zeno::vec3f(0.375f+u2*sc,v2*sc,0);
                uvw3=zeno::vec3f(0.375f+u3*sc,v3*sc,0);
                uvw4=zeno::vec3f(0.375f+u4*sc,v4*sc,0);

                // Bottom
                if(quad){
                    prim->loops.push_back(i1);
                    prim->loops.push_back(i2);
                    prim->loops.push_back(i4);
                    prim->loops.push_back(i3);
                    prim->polys.push_back({ss * 4, 4});
                }else{
                    indis.emplace_back(i1,i3,i2);
                    indis.emplace_back(i4,i2,i3);
                    uv1.push_back(uvw1);uv2.push_back(uvw3);uv3.push_back(uvw2);
                    uv1.push_back(uvw4);uv2.push_back(uvw2);uv3.push_back(uvw3);
                }
                ss++;

                int i1_,i2_,i3_,i4_;
                i1_=pp+i+(j+1)*pcircle;
                i2_=i1_+pcircle;
                if(j==div_w-3)
                    i2_=rls+i;
                i3_=i1_+1;
                i4_=i2_+1;

                uvw1=zeno::vec3f(0.375f+u1*sc,0.5f+(1.0f-v1)*sc,0);
                uvw2=zeno::vec3f(0.375f+u2*sc,0.5f+(1.0f-v2)*sc,0);
                uvw3=zeno::vec3f(0.375f+u3*sc,0.5f+(1.0f-v3)*sc,0);
                uvw4=zeno::vec3f(0.375f+u4*sc,0.5f+(1.0f-v4)*sc,0);

                if(j!=div_w-3&&i!=div_d-2){
                    nors[i4]=zeno::vec3f(0,-1,0);
                    nors[i4_]=zeno::vec3f(0,1,0);
                }

                // Top
                if(quad){
                    prim->loops.push_back(i3_);
                    prim->loops.push_back(i4_);
                    prim->loops.push_back(i2_);
                    prim->loops.push_back(i1_);
                    prim->polys.push_back({ss * 4, 4});
                }else{
                    indis.emplace_back(i1_,i2_,i3_);
                    indis.emplace_back(i4_,i3_,i2_);
                    uv1.push_back(uvw1);uv2.push_back(uvw2);uv3.push_back(uvw3);
                    uv1.push_back(uvw4);uv2.push_back(uvw3);uv3.push_back(uvw2);
                }
                ss++;
            }
        }


        for (int j = 0; j < div_w-1; j++)
        {
            for (int i = 0; i < div_h-1; i++)
            {
                int i1,i2,i3,i4,i1_,i2_,i3_,i4_;

                int tc=le+div_d+j*pcircle;
                int ci1=tc+i*2; // front and back point increase

                if(j==div_w-2 && i!=0){
                    i1=tc+i*div_d;  // depth increase
                    i2=i1-div_d;
                    i3=ci1-pcircle;
                }else{
                    i1=ci1;
                    i2=i1-2;
                    i3=i1-pcircle;
                }
                if(j==0){
                    i3=div_d*(i+1);
                    i4=i3-div_d;
                }else{
                    i4=i3-2;
                }

                if(i==0){
                    i2=i1-div_d;
                    i4=i3-div_d;
                }

                i1_=i1+1;
                i2_=i2+1;
                i3_=i3+1;
                i4_=i4+1;
                if(i==div_h-2 || j==div_w-2&&i!=div_h-2)
                    i1_=i1+(div_d-1);
                if(i==0 || j==div_w-2)
                    i2_=i2+(div_d-1);
                if(j!=0&&i==div_h-2)
                    i3_=i3+(div_d-1);
                if(i==0||j==0)
                    i4_=i4+(div_d-1);
                if(j==0)
                    i3_=i3+(div_d-1);

                float u1,v1,u2,v2,u3,v3,u4,v4;
                zeno::vec3f uvw1,uvw2,uvw3,uvw4;

                u4=j*sw;
                v4=i*sh;
                u3=u4;
                v3=v4+sh;
                u1=u4+sw;
                v1=v3;
                u2=u1;
                v2=v4;

                uvw1=zeno::vec3f(0.375f+u1*sc,0.25f+v1*sc,0);
                uvw2=zeno::vec3f(0.375f+u2*sc,0.25f+v2*sc,0);
                uvw3=zeno::vec3f(0.375f+u3*sc,0.25f+v3*sc,0);
                uvw4=zeno::vec3f(0.375f+u4*sc,0.25f+v4*sc,0);

                // Back
                if(quad){
                    prim->loops.push_back(i1);
                    prim->loops.push_back(i2);
                    prim->loops.push_back(i4);
                    prim->loops.push_back(i3);
                    prim->polys.push_back({ss * 4, 4});
                }else{
                    indis.emplace_back(i1,i3,i2);
                    indis.emplace_back(i4,i2,i3);
                    uv1.push_back(uvw1);uv2.push_back(uvw3);uv3.push_back(uvw2);
                    uv1.push_back(uvw4);uv2.push_back(uvw2);uv3.push_back(uvw3);
                }
                ss++;

                uvw1=zeno::vec3f(0.375f+u1*sc,0.75f+(1.0f-v1)*sc,0);
                uvw2=zeno::vec3f(0.375f+u2*sc,0.75f+(1.0f-v2)*sc,0);
                uvw3=zeno::vec3f(0.375f+u3*sc,0.75f+(1.0f-v3)*sc,0);
                uvw4=zeno::vec3f(0.375f+u4*sc,0.75f+(1.0f-v4)*sc,0);

                if(j!=div_w-2&&i!=div_h-2){
                    nors[i1]=zeno::vec3f(0,0,1);
                    nors[i1_]=zeno::vec3f(0,0,-1);
                }

                // Front
                if(quad){
                    prim->loops.push_back(i3_);
                    prim->loops.push_back(i4_);
                    prim->loops.push_back(i2_);
                    prim->loops.push_back(i1_);
                    prim->polys.push_back({ss * 4, 4});
                }else{
                    indis.emplace_back(i1_,i2_,i3_);
                    indis.emplace_back(i4_,i3_,i2_);
                    uv1.push_back(uvw1);uv2.push_back(uvw2);uv3.push_back(uvw3);
                    uv1.push_back(uvw4);uv2.push_back(uvw3);uv3.push_back(uvw2);
                }
                ss++;
            }
        }

        for (int i = 0; i < div_d; i++)
        {
            if(i==0){
                nors[i]=normalize(zeno::vec3f(0.3,-0.3,0.3));
                nors[i+inc]=normalize(zeno::vec3f(-0.3,-0.3,0.3));
                nors[i+rls]=normalize(zeno::vec3f(-0.3,0.3,0.3));
                nors[i+pp]=normalize(zeno::vec3f(0.3,0.3,0.3));
            }
            else if(i==div_d-1){
                nors[i]=normalize(zeno::vec3f(0.3,-0.3,-0.3));
                nors[i+inc]=normalize(zeno::vec3f(-0.3,-0.3,-0.3));
                nors[i+rls]=normalize(zeno::vec3f(-0.3,0.3,-0.3));
                nors[i+pp]=normalize(zeno::vec3f(0.3,0.3,-0.3));
            }
            else{
                nors[i]=normalize(zeno::vec3f(0.5,-0.5,0));
                nors[i+inc]=normalize(zeno::vec3f(-0.5,-0.5,0));
                nors[i+rls]=normalize(zeno::vec3f(-0.5,0.5,0));
                nors[i+pp]=normalize(zeno::vec3f(0.5,0.5,0));
            }
        }

        for (int i = 1; i < div_h-1; i++)
        {
            int i1=i*(div_d);
            int i2=i1+(div_d-1);
            int i3=inc+(i+1)*(div_d)-1;
            int i4=i3-(div_d-1);

            nors[i1]=normalize(zeno::vec3f(0.5,0,0.5));
            nors[i2]=normalize(zeno::vec3f(0.5,0,-0.5));
            nors[i3]=normalize(zeno::vec3f(-0.5,0,-0.5));
            nors[i4]=normalize(zeno::vec3f(-0.5,0,0.5));
        }

        for (int i = 0; i < div_w-2; i++)
        {
            int i1=le+i*pcircle;
            int i2=i1+(div_d-1);
            int i3=pcircle+pp+i*pcircle;
            int i4=i3+(div_d-1);

            nors[i1]=normalize(zeno::vec3f(0,-0.5,0.5));
            nors[i2]=normalize(zeno::vec3f(0,-0.5,-0.5));
            nors[i3]=normalize(zeno::vec3f(0,0.5,0.5));
            nors[i4]=normalize(zeno::vec3f(0,0.5,-0.5));
        }

        for (int i = 0; i < verts->size(); i++)
        {
            auto p = verts[i];
            auto n = nors[i];

            p = p * scale * size;

            ROTATE_COMPUTE
                p+= position;

            auto gn = glm::vec3(n[0], n[1], n[2]);
            gn = mz * my * mx * gn;
            n = zeno::vec3f(gn.x, gn.y, gn.z);

            verts[i] = p;
            nors[i] = n;
        }

        NORMUV_CIHOU
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(CreateCube, {
    {
        {"vec3f", "position", "0, 0, 0"},
        {"vec3f", "scaleSize", "1, 1, 1"},
        {"vec3f", "rotate", "0, 0, 0"},
        {"bool", "hasNormal", "0"},
        {"bool", "hasVertUV", "0"},
        {"bool", "isFlipFace", "0"},
        {"int", "div_w", "2", "X方向的切分数量"},
        {"int", "div_h", "2", "Y方向的切分数量"},
        {"int", "div_d", "2", "Z方向的切分数量"},
        {"float", "size", "1", "方块的大小"},
        {"bool", "quads", "0", "生成四边形网格"},
    },
    {"prim"},
    {},
    {"create"},
    {"创建一个立方体"},
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
            auto p4uv = p * scaleSize;
            p = p4uv;

            ROTATE_COMPUTE
                p+= position;

            verts.emplace_back(p);
            tris.emplace_back(i+1, 0, i+2);
            uv.emplace_back(p4uv[0]/2.0+0.5,
                            p4uv[2]/2.0+0.5, 0);
            norm.emplace_back(0, 1, 0);
        }

        // Update last
        tris[tris.size()-1] = zeno::vec3i(divisions, 0, 1);

        NORMUV_CIHOU
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(CreateDisk, {
    {
        {"vec3f", "position", "0, 0, 0"},
        {"vec3f", "scaleSize", "1, 1, 1"},
        {"vec3f", "rotate", "0, 0, 0"},
        {"bool", "hasNormal", "0"},
        {"bool", "hasVertUV", "0"},
        {"bool", "isFlipFace", "0"},
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
        auto rows = get_input2<int>("rows");
        auto columns = get_input2<int>("columns");
        auto quad = get_input2<bool>("quads");

        ROTATE_MATRIX

        auto &verts = prim->verts;
        auto &quads = prim->quads;
        auto &tris = prim->tris;
        auto &poly = prim->polys;
        auto &loops = prim->loops;

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
        float rm = 1.0f / rows;
        float cm = 1.0f / columns;
        int fi = 0;

        // Vertices & UV
        for(int i=0; i<=rows; i++){

            auto rp = start_point - zeno::vec3f(i*rm, 0, 0);

            for(int j=0; j<=columns; j++){
                auto p = rp - zeno::vec3f(0, 0, j*cm);
                p = p * scale * size;

                ROTATE_COMPUTE
                    p +=position;

                auto zcp = zeno::vec3f(p[0], p[1], p[2]);
                verts.push_back(zcp);
                uvs.emplace_back(i*rm, j*cm*-1+1, 0);
            }
        }

        // Indices
        int ss = 0;
        for(int i=0; i<rows; i++){
            for(int j=0; j<columns; j++){
                int i1 = fi;
                int i2 = i1+1;
                int i3 = fi+(columns+1);
                int i4 = i3+1;

                if(quad){
                    prim->loops.push_back(i1);
                    prim->loops.push_back(i2);
                    prim->loops.push_back(i4);
                    prim->loops.push_back(i3);
                    prim->polys.push_back({ss * 4, 4});
                }
                else{
                    tris.emplace_back(i1, i4, i2);
                    tris.emplace_back(i3, i4, i1);
                }
                ss++;
                fi += 1;
            }
            fi += 1;
        }

        // Normal
        for(int i=0; i<1; i++){
            vec3i ind;
            if(quad){
                ind = vec3i(loops[4*i],loops[4*i+1],loops[4*i+2]);
            }
            else{
                ind = tris[i];
            }
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
            uv[i] = zeno::vec3f(1 - uvs[i][0], 1 - uvs[i][1], 0);
            norm[i] = normal;
        }

        for (int i = 0; i < uvs.size(); i++) {
            prim->uvs.emplace_back(uvs[i][0], uvs[i][1]);
        }

        if(prim->loops.size()!= 0 && get_input2<bool>("hasVertUV")){
            loops.add_attr<int>("uvs");
            for (auto i = 0; i < prim->loops.size(); i++) {
                auto lo = prim->loops[i];
                loops.attr<int>("uvs")[i] = lo;
            }
        }

        prim->userData().setLiterial("pos", std::move(position));
        prim->userData().setLiterial("scale", std::move(scale));
        prim->userData().setLiterial("rotate", std::move(rotate));

        NORMUV_CIHOU
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(CreatePlane, {
    {
        {"vec3f", "position", "0, 0, 0"},
        {"vec3f", "scaleSize", "1, 1, 1"},
        {"vec3f", "rotate", "0, 0, 0"},
        {"bool", "hasNormal", "0"},
        {"bool", "hasVertUV", "0"},
        {"bool", "isFlipFace", "0"},
        {"float", "size", "1"},
        {"int", "rows", "1"},
        {"int", "columns", "1"},
        {"bool", "quads", "0"},
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
        auto &nors = prim->verts.add_attr<zeno::vec3f>("nrm");
        auto &uv1 = prim->tris.add_attr<zeno::vec3f>("uv0");
        auto &uv2 = prim->tris.add_attr<zeno::vec3f>("uv1");
        auto &uv3 = prim->tris.add_attr<zeno::vec3f>("uv2");

        if(rows <= 3)
            rows = 3;
        if(columns <= 3)
            columns = 3;

        std::vector<zeno::vec3f> mverts;

        float hs = height/2.0;

        verts.emplace_back(0,hs,0);
        nors.emplace_back(0,1,0);
        for (int i = 0; i < columns; i++) {
            uv1.emplace_back(0.5, 0.85, 0);
        }
        for (int i = 0; i < columns; i++) {
            float rad = 2.0f * M_PI * i / columns;
            float x = cos(rad);
            float z = -sin(rad);
            auto p1 = zeno::vec3f(x*radius1, hs, z*radius1);

            verts.push_back(p1);
            nors.emplace_back(0,1,0);
        }
        for (int i = 0; i < columns; i++) {
            float rad = 2.0f * M_PI * i / columns;
            float x = cos(rad);
            float z = -sin(rad);
            float of = 0.125;
            auto p1 = zeno::vec3f(x*radius2, -hs, z*radius2);

            verts.push_back(p1);
            nors.emplace_back(0,-1,0);
        }
        verts.emplace_back(0,-hs,0);
        nors.emplace_back(0,-1,0);
        for (int i = 0; i < columns; i++) {
            uv1.emplace_back(0.5, 0.15, 0);
        }

        for (int i = 1; i < columns+1; i++) {
            float rad = 2.0f * M_PI * (i) / columns;
            float x = cos(rad);
            float z = -sin(rad);
            float radn = 2.0f * M_PI * (i-1) / columns;
            float xn = cos(radn);
            float zn = -sin(radn);

            int i1,i2,i3;
            i1 = 0;
            i2 = i;
            i3 = i2+1;
            if(i == columns)
                i3 = 1;
            indis.emplace_back(i1, i3, i2);

            float u1=x*0.5*3/10+0.5;
            float v1=z*0.5*3/10+0.85;
            float u1n=xn*0.5*3/10+0.5;
            float v1n=zn*0.5*3/10+0.85;

            uv2.emplace_back(u1, v1, 0);
            uv3.emplace_back(u1n, v1n, 0);
        }

        for (int i = 1; i < columns+1; i++) {
            float rad = 2.0f * M_PI * (i-1) / columns;
            float x = cos(rad);
            float z = -sin(rad);
            float radn = 2.0f * M_PI * (i) / columns;
            float xn = cos(radn);
            float zn = -sin(radn);

            float b1,b2,b3;
            b1 = verts->size()-1;
            b2 = b1-columns+i-1;
            b3 = b2+1;
            if(i == columns)
                b3 -= columns;
            indis.emplace_back(b1, b2, b3);

            float u1=x*0.5*3/10+0.5;
            float v1=z*0.5*3/10+0.85;
            float u1n=xn*0.5*3/10+0.5;
            float v1n=zn*0.5*3/10+0.85;

            uv2.emplace_back(u1, v1-0.7, 0);
            uv3.emplace_back(u1n, v1n-0.7, 0);
        }

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                float f = float(i)/(rows-1);
                int ti = j+1;
                int bi = verts->size()-1-columns+j;
                auto pu = verts[ti];
                auto pd = verts[bi];
                auto np = (1-f)*pu+(f)*pd;

                if(i!=0 && i!=(rows-1)){
                    mverts.push_back(np);
                    nors.emplace_back(0,0,0);
                }
            }
        }

        for (int i = 0; i < mverts.size(); i++)
        {
            verts.push_back(mverts[i]);
        }

        for (int i = 1; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                int i1, i2, i3, i4;
                i1 = j+i*(columns)+2; //Two points at the top and bottom
                if(i==1)
                    i1-=columns+1;
                i2 = i1+1;
                i3 = i1+columns;
                if(i==1)
                    i3+=columns+1;
                i4 = i3+1;
                if(j == columns-1){
                    i2 -=columns;
                    i4 -=columns;
                }
                if(i==rows-1){
                    i3 -=i1-1-j;
                    i4 =i3+1;
                    if(j==columns-1)
                        i4-=columns;
                }

                float u1,v1,u2,v2,u3,v3,u4,v4;
                float sf = 4.0/10;
                float uf = 0.3;

                u1=float(j)/columns     *sf+uf;
                u2=float(j+1)/columns   *sf+uf;
                v1=float(i-1)/(rows-1)  *sf+uf;
                v2=v1;
                u3=u1;
                u4=u2;
                v3=float(i)/(rows-1)    *sf+uf;
                v4=v3;

                uv1.emplace_back(u1,v1,0);uv2.emplace_back(u2,v2,0);uv3.emplace_back(u4,v4,0);
                uv1.emplace_back(u1,v1,0);uv2.emplace_back(u4,v4,0);uv3.emplace_back(u3,v3,0);
                indis.emplace_back(i1, i2, i4);
                indis.emplace_back(i1, i4, i3);

                int b4=i1-1;
                if(j==0)
                    b4+=columns;
                auto p1 = verts[i1]*scale;
                auto p2 = verts[i2]*scale;
                auto p3 = verts[i3]*scale;
                auto p4 = verts[b4]*scale;
                auto n1 = normalize(cross(p2-p1, p1-p3));
                auto n2 = normalize(cross(p4-p1, p3-p1));

                zeno::vec3f n;
                n = normalize((n1+n2)/2.0);
                if(i==1){
                    auto up=zeno::vec3f(0,1,0);
                    nors[i1] = normalize((n1+n2+up)*scale/3.0);
                }
                if(i==rows-1){
                    auto down=zeno::vec3f(0,-1,0);
                    nors[i3] = normalize((n1+n2+down)*scale/3.0);
                    continue;
                }
                nors[i3] = n;
            }
        }

        for (int i = 0; i < verts->size(); i++)
        {
            auto p = verts[i];
            auto n = nors[i];


            p = p * scale ;

            ROTATE_COMPUTE
                p+=position;

            auto gn = glm::vec3(n[0], n[1], n[2]);
            gn = mz * my * mx * gn;
            n = zeno::vec3f(gn.x, gn.y, gn.z);

            verts[i] = p;
            nors[i] = n;
        }

        NORMUV_CIHOU
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(CreateTube, {
    {
        {"vec3f", "position", "0, 0, 0"},
        {"vec3f", "scaleSize", "1, 1, 1"},
        {"vec3f", "rotate", "0, 0, 0"},
        {"bool", "hasNormal", "0"},
        {"bool", "hasVertUV", "0"},
        {"bool", "isFlipFace", "0"},
        {"float", "radius1", "1"},
        {"float", "radius2", "1"},
        {"float", "height", "2"},
        {"int", "rows", "3"},
        {"int", "columns", "12"}
    },
    {"prim"},
    {},
    {"create"},
});

struct CreateTorus : zeno::INode {
    virtual void apply() override {
        auto majorSegment = get_input2<int>("MajorSegment");
        auto minorSegment = get_input2<int>("MinorSegment");
        auto majorRadius = get_input2<float>("MajorRadius");
        auto minorRadius = get_input2<float>("MinorRadius");

        if (majorSegment < 3) {
            majorSegment = 3;
        }
        if (minorSegment < 3) {
            minorSegment = 3;
        }

        auto prim = std::make_shared<zeno::PrimitiveObject>();
        prim->verts.resize(majorSegment * minorSegment);
        auto &nrm = prim->verts.add_attr<zeno::vec3f>("nrm");
        for (auto j = 0; j < minorSegment; j++) {
            float theta = M_PI * 2.0 * j / minorSegment - M_PI;
            float y = sin(theta) * minorRadius;
            auto radius = majorRadius + cos(theta) * minorRadius;
            for (auto i = 0; i < majorSegment; i++) {
                int index = j * majorSegment + i;
                float phi = M_PI * 2.0 * i / majorSegment;
                vec3f pos = {cos(phi) * radius, y, sin(phi) * radius};
                vec3f refCenter = {cos(phi) * majorRadius, 0, sin(phi) * majorRadius};
                prim->verts[index] = pos;
                nrm[index] = zeno::normalize(pos - refCenter);
            }
        }

        prim->uvs.resize((majorSegment + 1) * (minorSegment + 1));
        for (auto j = 0; j < minorSegment + 1; j++) {
            for (auto i = 0; i < majorSegment + 1; i++) {
                int index = j * (majorSegment + 1) + i;
                prim->uvs[index] = { float(j) / (majorSegment + 1), float(i) / (minorSegment + 1) };
            }
        }

        prim->loops.resize(minorSegment * majorSegment * 4);
        auto &uvs = prim->loops.add_attr<int>("uvs");
        for (auto j = 0; j < minorSegment; j++) {
            for (auto i = 0; i < majorSegment; i++) {
                int index = j * majorSegment + i;
                prim->loops[index * 4 + 0] = j * majorSegment + i;
                prim->loops[index * 4 + 1] = ((j + 1) * majorSegment + i) % (minorSegment * majorSegment);
                prim->loops[index * 4 + 2] = ((j + 1) * majorSegment + (i + 1) % majorSegment) % (minorSegment * majorSegment);
                prim->loops[index * 4 + 3] = j * majorSegment + (i + 1) % majorSegment;
                uvs[index * 4 + 0] = j * (majorSegment + 1) + i;
                uvs[index * 4 + 1] = (j + 1) * (majorSegment + 1) + i;
                uvs[index * 4 + 2] = (j + 1) * (majorSegment + 1) + i + 1;
                uvs[index * 4 + 3] = j * (majorSegment + 1) + i + 1;
            }
        }

        prim->polys.resize(minorSegment * majorSegment);
        for (auto i = 0; i < prim->polys.size(); i++) {
            prim->polys[i] = {i * 4, 4};
        }
        auto position = get_input2<zeno::vec3f>("position");
        auto rotate = get_input2<zeno::vec3f>("rotate");
        glm::mat4 transform = glm::mat4 (1.0);
        transform = glm::translate(transform, glm::vec3(position[0], position[1], position[2]));

            auto order = get_input2<std::string>("EulerRotationOrder:");
            auto orderTyped = magic_enum::enum_cast<EulerAngle::RotationOrder>(order).value_or(EulerAngle::RotationOrder::YXZ);

            auto measure = get_input2<std::string>("EulerAngleMeasure:");
            auto measureTyped = magic_enum::enum_cast<EulerAngle::Measure>(measure).value_or(EulerAngle::Measure::Radians);

            glm::vec3 eularAngleXYZ = glm::vec3(rotate[0], rotate[1], rotate[2]);
            glm::mat4 rotation = EulerAngle::rotate(orderTyped, measureTyped, eularAngleXYZ);

        transform = transform * rotation;

        auto n_transform = glm::transpose(glm::inverse(transform));
        for(int i = 0; i < prim->verts.size(); i++){
            auto p = prim->verts[i];
            auto gp = transform * glm::vec4(p[0], p[1], p[2], 1);
            prim->verts[i] = zeno::vec3f(gp.x, gp.y, gp.z);
            auto n = nrm[i];
            auto gn = n_transform * glm::vec4 (n[0], n[1], n[2], 0);
            nrm[i] = zeno::vec3f (gn.x, gn.y, gn.z);
        }

        if (!get_input2<bool>("hasNormal")){
            prim->verts.attrs.erase("nrm");
        }

        if (!get_input2<bool>("hasVertUV")){
            prim->uvs.clear();
            prim->loops.erase_attr("uvs");
        }

        if (!get_input2<bool>("quads")){
            primTriangulate(prim.get());
        }
        set_output("prim",std::move(prim));
    }
};

ZENDEFNODE(CreateTorus, {
{
        {"vec3f", "position", "0, 0, 0"},
        {"vec3f", "rotate", "0, 0, 0"},
        {"float", "MajorRadius", "1"},
        {"float", "MinorRadius", "0.25"},
        {"bool", "hasNormal", "0"},
        {"bool", "hasVertUV", "0"},
        {"int", "MajorSegment", "48"},
        {"int", "MinorSegment", "12"},
        {"bool", "quads", "0"},
    },
    {"prim"},
    {
        {"enum " + EulerAngle::RotationOrderListString(), "EulerRotationOrder", "XYZ"},
        {"enum " + EulerAngle::MeasureListString(), "EulerAngleMeasure", "Degree"}
    },
    {"create"},
});

struct CreateSphere : zeno::INode {
    virtual void apply() override {
        auto prim = std::make_shared<zeno::PrimitiveObject>();
        auto position = get_input2<zeno::vec3f>("position");
        auto scale = get_input2<zeno::vec3f>("scaleSize");
        auto rotate = get_input2<zeno::vec3f>("rotate");
        auto rows = get_input2<int>("rows");
        auto columns = get_input2<int>("columns");
        auto radius = get_input2<float>("radius");
        auto quad = get_input2<bool>("quads");

        if (rows < 2) {
            rows = 2;
        }
        if (columns < 3) {
            columns = 3;
        }
        std::vector<zeno::vec3f> verts = {};
        std::vector<zeno::vec2i> poly = {};
        std::vector<int> loops = {};
        std::vector<zeno::vec2f> uvs = {};
        std::vector<zeno::vec3f> nors = {};

        verts.push_back (vec3f(0,1,0));
        for (auto row = 1; row < rows; row++) {
            float v = (float)row / (float)rows;
            float theta = M_PI * v;
            for (auto column = 0; column < columns; column++) {
                float u = (float)column / (float)columns;
                float phi = M_PI * 2 * u;
                float x = sin(theta) * cos(phi);
                float y = cos(theta);
                float z = -sin(theta) * sin(phi);
                verts.push_back(vec3f(x,y,z));
            }
        }
        verts.push_back (vec3f(0,-1,0));
        {
            //head
            for (auto column = 0; column < columns; column++) {
                if (column == columns - 1) {
                    loops.push_back(0);
                    loops.push_back(columns);
                    loops.push_back(1);
                    poly.push_back(vec2i(column * 3, 3));
                } else {
                    loops.push_back(0);
                    loops.push_back(column + 1);
                    loops.push_back(column + 2);
                    poly.push_back(vec2i(column * 3, 3));
                }
            }
            //body
            for (auto row = 1; row < rows - 1; row++) {
                for (auto column = 0; column < columns; column++) {
                    if (column == columns - 1) {
                        loops.push_back((row - 1) * columns + 1);
                        loops.push_back((row - 1) * columns + columns);
                        loops.push_back(row * columns + columns);
                        loops.push_back(row * columns + 1);
                        poly.push_back(vec2i(columns * 3 + (row - 1) * columns * 4 + column * 4, 4));
                    } else {
                        loops.push_back((row - 1) * columns + column + 2);
                        loops.push_back((row - 1) * columns + column + 1);
                        loops.push_back(row * columns + column + 1);
                        loops.push_back(row * columns + column + 2);
                        poly.push_back(vec2i(loops.size() - 4, 4));
                    }
                }
            }
            //tail
            for (auto column = 0; column < columns; column++) {
                if (column == columns - 1) {
                    loops.push_back((rows - 2) * columns + 1);
                    loops.push_back((rows - 2) * columns + column + 1);
                    loops.push_back((rows - 1) * columns + 1);
                    poly.push_back(vec2i(columns * 3 + (rows - 2) * columns * 4 + column * 3, 3));
                } else {
                    loops.push_back((rows - 2) * columns + column + 2);
                    loops.push_back((rows - 2) * columns + column + 1);
                    loops.push_back((rows - 1) * columns + 1);
                    poly.push_back(vec2i(loops.size() - 3, 3));
                }
            }
        }

        for(int column = 0;column < columns;column++){
            uvs.push_back({(column+0.5f)/columns, 1.0f, 0.0f});
        }
        for(int row = 1;row < rows;row++){
            for(int column = 0;column < columns+1;column++){
                uvs.push_back({(column+0.0f)/columns,1.0f-(row+0.0f)/rows,0.0f});
            }
        }
        for(int column = 0;column < columns;column++){
            uvs.push_back({(column+0.5f)/columns, 0.0f, 0.0f});
        }

        auto& loops_uv = prim->loops.add_attr<int>("uvs");
        loops_uv.resize(0);
        for(int column = 0;column < columns;column++){
            loops_uv.push_back(column);
            loops_uv.push_back(columns+column);
            loops_uv.push_back(columns+column+1);
        }
        for(int row = 1;row < rows-1;row++){
            for(int column = 0;column < columns;column++){
                loops_uv.push_back(columns+(columns+1)*(row-1)+column+1);
                loops_uv.push_back(columns+(columns+1)*(row-1)+column);
                loops_uv.push_back(columns+(columns+1)*row+column);
                loops_uv.push_back(columns+(columns+1)*row+column+1);
            }
        }
        for(int column = 0;column < columns;column++){
            loops_uv.push_back(columns+(columns+1)*(rows-2)+column+1);
            loops_uv.push_back(columns+(columns+1)*(rows-2)+column);
            loops_uv.push_back(columns+(columns+1)*(rows-1)+column);
        }

        glm::mat4 transform = glm::mat4 (1.0);
        transform = glm::translate(transform, glm::vec3(position[0], position[1], position[2]));

            auto order = get_input2<std::string>("EulerRotationOrder:");
            auto orderTyped = magic_enum::enum_cast<EulerAngle::RotationOrder>(order).value_or(EulerAngle::RotationOrder::YXZ);

            auto measure = get_input2<std::string>("EulerAngleMeasure:");
            auto measureTyped = magic_enum::enum_cast<EulerAngle::Measure>(measure).value_or(EulerAngle::Measure::Radians);

            glm::vec3 eularAngleXYZ = glm::vec3(rotate[0], rotate[1], rotate[2]);
            glm::mat4 rotation = EulerAngle::rotate(orderTyped, measureTyped, eularAngleXYZ);

        transform = transform * rotation;
 
        transform = glm::scale(transform, glm::vec3(scale[0],scale[1],scale[2]) * radius);

        auto n_transform = glm::transpose(glm::inverse(transform));

        nors.resize(verts.size());
        for(int i = 0; i < verts.size(); i++){
            auto n = verts[i];
            auto p = verts[i];
            auto gp = transform * glm::vec4(p[0], p[1], p[2], 1);
            verts[i] = zeno::vec3f(gp.x, gp.y, gp.z);
            auto gn = n_transform * glm::vec4 (n[0], n[1], n[2], 0);
            nors[i] = zeno::vec3f (gn.x, gn.y, gn.z);
        }

        prim->verts.resize(verts.size());
        for (auto i = 0; i < verts.size(); i++) {
            prim->verts[i] = verts[i];
        }
        prim->polys.resize(poly.size());
        for (auto i = 0; i < poly.size(); i++) {
            prim->polys[i] = poly[i];
        }
        prim->loops.resize(loops.size());
        for (auto i = 0; i < loops.size(); i++) {
            prim->loops[i] = loops[i];
        }
        prim->uvs.resize(uvs.size());

        for (auto i = 0; i < uvs.size(); i++) {
            prim->uvs[i] = uvs[i];
        }

        auto &nors2 = prim->verts.add_attr<zeno::vec3f>("nrm");
        for (auto i = 0; i < nors.size(); i++) {
            nors2[i] = nors[i];
        }

        if (!get_input2<bool>("hasNormal")){
            prim->verts.attrs.erase("nrm");
        }

        if (!get_input2<bool>("hasVertUV")){
            prim->uvs.clear();
            prim->loops.erase_attr("uvs");
        }

        if (get_input2<bool>("isFlipFace")){
            for (auto i = 0; i < prim->polys.size(); i++) {
                auto [base, cnt] = prim->polys[i];
                for (int j = 0; j < (cnt / 2); j++) {
                    std::swap(prim->loops[base + j], prim->loops[base + cnt - 1 - j]);
                    if (prim->loops.has_attr("uvs")) {
                        std::swap(prim->loops.attr<int>("uvs")[base + j], prim->loops.attr<int>("uvs")[base + cnt - 1 - j]);
                    }
                }
            }
        }

        if(!quad){
            primTriangulate(prim.get());
        }

        auto SphereRT = get_input2<bool>("SphereRT");

        if (SphereRT) {
            prim->userData().set2("sphere_center", std::move(position));
            prim->userData().set2("sphere_radius", std::move(radius));

            prim->userData().set2("sphere_rotate", std::move(rotate));
            prim->userData().set2("sphere_scale", std::move(scale));

            // auto sphere_transform = std::make_shared<zeno::MatrixObject>();
            // sphere_transform->m = transform;
            auto transform_ptr = glm::value_ptr(transform);
            
            zeno::vec4f row0, row1, row2, row3;
            //zeno::vec<16, float> tmp_array;
            memcpy(row0.data(), transform_ptr, sizeof(float)*4);
            memcpy(row1.data(), transform_ptr+4, sizeof(float)*4);
            memcpy(row2.data(), transform_ptr+8, sizeof(float)*4);  
            memcpy(row3.data(), transform_ptr+12, sizeof(float)*4);

            prim->userData().set2("_transform_row0", row0);
            prim->userData().set2("_transform_row1", row1);
            prim->userData().set2("_transform_row2", row2);
            prim->userData().set2("_transform_row3", row3);
        }

        set_output("prim",std::move(prim));
    }
};

ZENDEFNODE(CreateSphere, {
    {
        {"vec3f", "position", "0, 0, 0"},
        {"vec3f", "scaleSize", "1, 1, 1"},
        {"float", "radius", "1"},
        {"vec3f", "rotate", "0, 0, 0"},
        {"bool", "hasNormal", "0"},
        {"bool", "hasVertUV", "0"},
        {"bool", "isFlipFace", "0"},
        {"int", "rows", "12"},
        {"int", "columns", "24"},
        {"bool", "quads", "0"},
        {"bool", "SphereRT", "0"}
    },
    {"prim"},
    {
        {"enum " + EulerAngle::RotationOrderListString(), "EulerRotationOrder", "XYZ"},
        {"enum " + EulerAngle::MeasureListString(), "EulerAngleMeasure", "Degree"}
    },
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
struct CreateFolder : zeno::INode {
    virtual void apply() override {
        namespace fs = std::filesystem;
        auto folderPath = fs::u8path(get_input2<std::string>("folderPath"));
        if (!fs::exists(folderPath)) {
            fs::create_directories(folderPath);
        }
    }
};

ZENDEFNODE(CreateFolder, {
    {
        {"directory", "folderPath"}
    },
    {},
    {},
    {"create"},
});

struct RemoveFolder : zeno::INode {
    virtual void apply() override {
        namespace fs = std::filesystem;
        auto folderPath = fs::u8path(get_input2<std::string>("folderPath"));
        if (fs::exists(folderPath)) {
            std::error_code errorCode;
            fs::remove_all(folderPath, errorCode);
            if (get_input2<bool>("clean")) {
                fs::create_directories(folderPath);
            }
        }
    }
};

ZENDEFNODE(RemoveFolder, {
    {
        {"directory", "folderPath"},
        {"bool", "clean", "false"},
    },
    {},
    {},
    {"create"},
});

struct FFMPEGImagesToVideo : zeno::INode {
    virtual void apply() override {
        namespace fs = std::filesystem;
        auto fps = get_input2<int>("fps");
        auto imageFolderPath = get_input2<std::string>("imageFolderPath");
        auto bitrate = get_input2<int>("bitrate");
        auto outPath = get_input2<std::string>("outPath");

        bool ok = fs::exists(imageFolderPath) && fs::is_directory(imageFolderPath);
        if (!ok) {
            throw zeno::makeError("imageFolderPath not exists or not is_directory");
        }
        std::vector<fs::path> filenames;
        std::string extension;
        for (const auto& entry : fs::directory_iterator(imageFolderPath)) {
            if (fs::is_regular_file(entry.status())) {
                filenames.emplace_back(entry.path().filename());
                extension = entry.path().filename().extension().string();
            }
        }
        std::sort(filenames.begin(), filenames.end());
        for (auto i = 0; i < filenames.size(); i++) {
            auto old_name = zeno::format("{}/{}", imageFolderPath, filenames[i].string());
            auto new_name = zeno::format("{}/{:07d}{}", imageFolderPath, i, extension);
            fs::rename(old_name, new_name);
        }

        auto cmd = zeno::format("ffmpeg -y -r {} -i {}/%07d{} -b:v {}k -c:v mpeg4 {}", fps, imageFolderPath, extension, bitrate, outPath);
        std::system(cmd.c_str());
    }
};

ZENDEFNODE(FFMPEGImagesToVideo, {
    {
        {"int", "fps", "25"},
        {"directory", "imageFolderPath", "imageFolderPath"},
        {"int", "bitrate", "200000"},
        {"writepath", "outPath", "outPath"},
    },
    {},
    {},
    {"Miscellaneous"},
});
}
}
