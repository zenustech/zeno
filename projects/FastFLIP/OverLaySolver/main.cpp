//
// Created by zhxx on 2021/1/4.
//


#include <cfloat>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "../FLIP/Sparse_buffer.h"
//#include "../FLIP/volumeMeshTools.h"
#include "OverLaySolver.h"
#include "../Parser/scene_parser.h"
using TV = FLUID::Vec3f;
using IV = FLUID::Vec3i;

float vdb_phi(const FLUID::Vec3f &position) {
    float phi = 1.f;
    for (auto &grid : boundaries) {
        openvdb::tools::GridSampler<TreeT, openvdb::tools::BoxSampler> interpolator(
                grid->constTree(), grid->transform());
        openvdb::math::Vec3<float> P(position[0], position[1], position[2]);
        auto tmp = interpolator.wsSample(P); // ws denotes world space
        if (tmp < phi)
            phi = tmp;
        if (phi < 0.f)
            break;
    }
    return (float)phi;
}

int main(int argc, char **argv) {
    Aero sim;
    openvdb::initialize();
    parse_scene(argv[1], sim);
    string outpath(simConfigs.outputDir);
    sim.setGravity(simConfigs.g);
    int frame = 0;
    for (double T=0;T<simConfigs.T;T+=simConfigs.dt) {
        printf("--------------------\nFrame %d\n", frame);

        sim.advance(simConfigs.dt, vdb_phi);

        //vdbToolsWapper::outputBgeo(outpath, frame, sim.otracers);
        //vdbToolsWapper::outputBin(outpath, frame, sim.particles);
        for(auto d:sim.domains)
        {
            d->eulerian_fluids.write_bulk_bgeo(outpath, frame+1);
        }
        printf("Exporting particle data\n");
        frame++;
    }
    return 0;
}