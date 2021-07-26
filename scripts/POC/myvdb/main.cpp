#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cmath>
#include <array>
#include "vec.h"
#include "Grid.h"
#include "HashGrid.h"
#include "LeafPoints.h"

int main() {
    Grid<HashGrid<Points<int, 32>>> grid, new_grid;

    const int L = grid.LeafShift;
    const int N = 3;
    std::vector<vec3f> pos(N);
    std::vector<vec3f> new_pos(N);
    std::vector<vec3f> vel(N);
    for (int i = 0; i < N; i++) {
        pos[i] = (vec3f)vec3i(
                rand(), rand(), rand()) / (float)RAND_MAX;
        vel[i] = (vec3f)vec3i(
                rand(), rand(), rand()) / (float)RAND_MAX * 2.f - 1.f;
    }

    grid.leaf_size = 0.04f;
    new_grid.leaf_size = 0.04f;

    float dt = 0.01f;

    // p2g
    for (int i = 0; i < N; i++) {
        auto iipos = pos[i] / grid.leaf_size;
        auto ipos = vec3i(iipos);
        auto jpos = vec3i(fmod(iipos, 1.f) * float(1 << L));
        Coord leafCoord{ipos[0], ipos[1], ipos[2]};
        auto *leaf = grid.leafAt(leafCoord);
        Coord subCoord{jpos[0], jpos[1], jpos[2]};
        leaf->insertElement(subCoord, i);
        //printf("? %d %d %d\n", leafCoord[0], leafCoord[1], leafCoord[2]);
        //printf("! %d %d %d\n", subCoord[0], subCoord[1], subCoord[2]);
    }

    // advect
    grid.foreachLeaf([&] (auto *leaf, Coord const &leafCoord) {
        leaf->foreachElement([&] (auto &value, int index) {
            Coord subCoord = leaf->indexToCoord(index);
            auto vel_dt = vel[value] * dt;
            subCoord[0] += int(vel_dt[0] * ((1 << L) / grid.leaf_size));
            subCoord[1] += int(vel_dt[1] * ((1 << L) / grid.leaf_size));
            subCoord[2] += int(vel_dt[2] * ((1 << L) / grid.leaf_size));

            Coord newLeafCoord = leafCoord;
            newLeafCoord[0] += subCoord[0] >> L;
            newLeafCoord[1] += subCoord[1] >> L;
            newLeafCoord[2] += subCoord[2] >> L;

            subCoord[0] &= (1 << L) - 1;
            subCoord[1] &= (1 << L) - 1;
            subCoord[2] &= (1 << L) - 1;

            //printf("? %d %d %d\n", newLeafCoord[0], newLeafCoord[1], newLeafCoord[2]);
            //printf("! %d %d %d\n", subCoord[0], subCoord[1], subCoord[2]);
            new_grid.leafAt(newLeafCoord)->insertElement(subCoord, value);
        });
    });

    // g2p
    new_grid.foreachLeaf([&] (auto *leaf, Coord const &leafCoord) {
        //printf("? %d %d %d\n", leafCoord[0], leafCoord[1], leafCoord[2]);
        leaf->foreachElement([&] (auto &value, int index) {
            Coord subCoord = leaf->indexToCoord(index);
            //printf("! %d %d %d\n", subCoord[0], subCoord[1], subCoord[2]);
            float fx = (leafCoord[0] + subCoord[0] * (1.f / (1 << L))) * grid.leaf_size;
            float fy = (leafCoord[1] + subCoord[1] * (1.f / (1 << L))) * grid.leaf_size;
            float fz = (leafCoord[2] + subCoord[2] * (1.f / (1 << L))) * grid.leaf_size;

            new_pos[value] = vec3f(fx, fy, fz);
        });
    });

    for (int i = 0; i < N; i++) {
        printf("%f %f\n", new_pos[i][0] - pos[i][0], vel[i][0] * dt);
        printf("%f %f\n", new_pos[i][1] - pos[i][1], vel[i][1] * dt);
        printf("%f %f\n", new_pos[i][2] - pos[i][2], vel[i][2] * dt);
    }
}
//for (int dz = 0; dz < 2; dz++) for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++) {
