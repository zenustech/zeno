#pragma once
#include <zen/zen.h>
#include <zen/MeshObject.h>
#include <zen/PrimitiveObject.h>
#include <zen/NumericObject.h>
#include <zen/ParticlesObject.h>
#include <glm/glm.hpp>
#include <glm/vec3.hpp>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include "tbb/scalable_allocator.h"
#include "tbb/concurrent_vector.h"
#include "tbb/parallel_for.h"
#include "iostream"

namespace zenbase {
//todo where to put this func???
float area(glm::vec3 &p1, glm::vec3 &p2, glm::vec3 &p3)
{
    glm::vec3 e1 = p3-p1;
    glm::vec3 e2 = p2-p1;
    glm::vec3 areavec = glm::cross(e1,e2);
    return 0.5*sqrt(glm::dot(areavec, areavec));
}
//todo where to put this func????
bool ptInTriangle(glm::vec3 &p, glm::vec3 &p0, glm::vec3 &p1, glm::vec3 &p2)
{
    float A = 0.5*(-p1.y * p2.x + p0.y * (-p1.x + p2.x) + p0.x * (p1.y - p2.y) + p1.x * p2.y);
    float sign = A < 0 ? -1.0f : 1.0f;
    float s = (p0.y * p2.x - p0.x * p2.y + (p2.y - p0.y) * p.x + (p0.x - p2.x) * p.y) * sign;
    float t = (p0.x * p1.y - p0.y * p1.x + (p0.y - p1.y) * p.x + (p1.x - p0.x) * p.y) * sign;
    
    return s > 0 && t > 0 && (s + t) < 2 * A * sign;
}

//to do where to put this func??
template<class T>
T baryCentricInterpolation(T &v1, T &v2, T &v3, glm::vec3 &p, glm::vec3 &vert1, glm::vec3 &vert2, glm::vec3 &vert3)
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
        tbb::concurrent_vector<glm::vec3> pos(0);
        tbb::concurrent_vector<glm::vec3> vel(0);
        size_t n = prim->triangles.size();
        printf("%d\n",n);
        std::cout<<channel<<std::endl;
        tbb::parallel_for((size_t)0, (size_t)n, (size_t)1, [&](size_t index)
            {
                glm::vec3 a, b, c;
                glm::ivec3 vi = prim->triangles[index];
                a = prim->attr<glm::vec3>("pos")[vi[0]];
                b = prim->attr<glm::vec3>("pos")[vi[1]];
                c = prim->attr<glm::vec3>("pos")[vi[2]];
                glm::vec3 e1 = b-a;
                glm::vec3 e2 = c-a;
                glm::vec3 dir1 = glm::normalize(e1);
                glm::vec3 dir2 = glm::normalize(e2);
                int in = glm::length(e1)/(0.5*dx);
                int jn = glm::length(e2)/(0.5*dx);
                glm::vec3 vel1 = prim->attr<glm::vec3>(channel)[vi[0]];
                glm::vec3 vel2 = prim->attr<glm::vec3>(channel)[vi[1]];
                glm::vec3 vel3 = prim->attr<glm::vec3>(channel)[vi[2]];
                for(int ii=0;ii<in;ii++)
                {
                    for(int jj=0;jj<jn;jj++)
                    {
                        glm::vec3 vij = a + (float)ii*0.5f*dx*dir1 + (float)jj*0.5f*dx*dir2;
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
                result->pos[index] = pos[index];
                result->vel[index] = vel[index];
            }
        );
        set_output("particles", result);
    }
};

static int defMeshToPrimitive = zen::defNodeClass<SprayParticles>("SprayParticles",
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

