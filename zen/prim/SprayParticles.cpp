#include <zen/zen.h>
#include <zen/MeshObject.h>
#include <zen/PrimitiveObject.h>
#include <zen/NumericObject.h>
#include <zen/ParticlesObject.h>
#include <zen/vec.h>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include "tbb/scalable_allocator.h"
#include "tbb/concurrent_vector.h"
#include "tbb/parallel_for.h"
#include "iostream"

namespace zenbase {
//todo where to put this func???
float area(zen::vec3f &p1, zen::vec3f &p2, zen::vec3f &p3)
{
    zen::vec3f e1 = p3-p1;
    zen::vec3f e2 = p2-p1;
    zen::vec3f areavec = zen::cross(e1,e2);
    return 0.5*sqrt(zen::dot(areavec, areavec));
}
//todo where to put this func????
bool ptInTriangle(zen::vec3f &p, zen::vec3f &p0, zen::vec3f &p1, zen::vec3f &p2)
{
    float A = 0.5*(-p1[1] * p2[0] + p0[1] * (-p1[0] + p2[0]) + p0[0] * (p1[1] - p2[1]) + p1[0] * p2[1]);
    float sign = A < 0 ? -1.0f : 1.0f;
    float s = (p0[1] * p2[0] - p0[0] * p2[1] + (p2[1] - p0[1]) * p[0] + (p0[0] - p2[0]) * p[1]) * sign;
    float t = (p0[0] * p1[1] - p0[1] * p1[0] + (p0[1] - p1[1]) * p[0] + (p1[0] - p0[0]) * p[1]) * sign;
    
    return s > 0 && t > 0 && (s + t) < 2 * A * sign;
}

//to do where to put this func??
template<class T>
T baryCentricInterpolation(T &v1, T &v2, T &v3, zen::vec3f &p, zen::vec3f &vert1, zen::vec3f &vert2, zen::vec3f &vert3)
{
    float a1 = area(p, vert2, vert3);
    float a2 = area(p, vert1, vert3);
    float a  = area(vert1,vert2,vert3);
    float w1 = a1/a;
    float w2 = a2/a;
    float w3 = 1 - w1 - w2;
    return w1 * v1 + w2 * v2 + w3 * v3;
}

struct SprayParticles : zen::INode{
    virtual void apply() override {
        auto dx = std::get<float>(get_param("dx"));
        auto channel = std::get<std::string>(get_param("channel"));
        auto prim = get_input("TrianglePrim")->as<PrimitiveObject>();
        auto result = zen::IObject::make<ParticlesObject>();
        tbb::concurrent_vector<zen::vec3f> pos(0);
        tbb::concurrent_vector<zen::vec3f> vel(0);
        size_t n = prim->triangles.size();
        printf("%d\n",n);
        std::cout<<channel<<std::endl;
        tbb::parallel_for((size_t)0, (size_t)n, (size_t)1, [&](size_t index)
            {
                zen::vec3f a, b, c;
                zen::vec3i vi = prim->triangles[index];
                a = prim->attr<zen::vec3f>("pos")[vi[0]];
                b = prim->attr<zen::vec3f>("pos")[vi[1]];
                c = prim->attr<zen::vec3f>("pos")[vi[2]];
                zen::vec3f e1 = b-a;
                zen::vec3f e2 = c-a;
                zen::vec3f dir1 = zen::normalize(e1);
                zen::vec3f dir2 = zen::normalize(e2);
                int in = zen::length(e1)/(0.5*dx);
                int jn = zen::length(e2)/(0.5*dx);
                zen::vec3f vel1 = prim->attr<zen::vec3f>(channel)[vi[0]];
                zen::vec3f vel2 = prim->attr<zen::vec3f>(channel)[vi[1]];
                zen::vec3f vel3 = prim->attr<zen::vec3f>(channel)[vi[2]];
                for(int ii=0;ii<in;ii++)
                {
                    for(int jj=0;jj<jn;jj++)
                    {
                        zen::vec3f vij = a + (float)ii*0.5f*dx*dir1 + (float)jj*0.5f*dx*dir2;
                        if(ptInTriangle(vij, a, b, c))
                        {
                            pos.emplace_back(vij);
                            vel.emplace_back(baryCentricInterpolation(vel1,vel2,vel3, vij, a,b,c));
                        }
                    }
                }

            }
        );
        result->pos.resize(pos.size());
        result->vel.resize(vel.size());
        tbb::parallel_for((size_t)0, (size_t)pos.size(), (size_t)1, [&](size_t index)
            {
                result->pos[index] = zen::vec_to_other<glm::vec3>(pos[index]);
                result->vel[index] = zen::vec_to_other<glm::vec3>(vel[index]);
            }
        );
        set_output("particles", result);
    }
};

static int defSprayParticles = zen::defNodeClass<SprayParticles>("SprayParticles",
    { /* inputs: */ {
        "TrianglePrim",
    }, /* outputs: */ {
        "particles",
    }, /* params: */ { 
        {"float", "dx", "0.01"},
        {"string", "channel", "vel"},
    }, /* category: */ {
        "primitive",
    }});

}

