//
// Created by zhxx on 2020/11/12.
//

#ifndef FLIPBEM_FLIPPARTICLE_H
#define FLIPBEM_FLIPPARTICLE_H
#include "util.h"
#include "vec.h"

struct FLIP_particle {
    FLUID::Vec3f pos;
    FLUID::Vec3f vort;
    FLUID::Vec3f vel;
    float length;
    float circ;
    float volm=1.0;
    FLIP_particle() {
        length = 0;
        for (int i = 0; i < 3; i++) {
            pos[i] = 0;
            vel[i] = 0;
            vort[i] = 0;
        }
        circ = 0;
        volm = 1;
    }
    ~FLIP_particle() {}
    FLIP_particle(const FLIP_particle &p) {
        pos = p.pos;
        vel = p.vel;
        vort = p.vort;
        length = p.length;
        circ = p.circ;
        volm = p.volm;
    }
    FLIP_particle(const FLUID::Vec3f &p, const FLUID::Vec3f &v) {
        pos = p;
        vel = v;
    }
    FLIP_particle(const FLUID::Vec3f &p, const FLUID::Vec3f &v,
                  const FLUID::Vec3f &w, const float &l, const float &_volm) {
        pos = p;
        vel = v;
        vort = w;
        length = l;
        circ = mag(vort) / length;
        volm = _volm;
    }
};
#endif //FLIPBEM_FLIPPARTICLE_H
