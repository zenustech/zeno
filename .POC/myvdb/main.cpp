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

using namespace fdb;

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
        auto leafCoord = vec3i(iipos);
        auto subCoord = vec3i(fmod(iipos, 1.f) * float(1 << L));
        auto *leaf = grid.leafAt(leafCoord);
        leaf->insertElement(subCoord, i);
    }

    // advect
    grid.foreachLeaf([&] (auto *leaf, Coord const &leafCoord) {
        leaf->foreachElement([&] (auto &value, int index) {
            Coord subCoord = leaf->indexToCoord(index);
            auto vel_dt = vel[value] * dt;
            subCoord += toint(vel_dt * ((1 << L) / grid.leaf_size));

            Coord newLeafCoord = leafCoord;
            newLeafCoord += subCoord >> L;
            subCoord &= (1 << L) - 1;

            new_grid.leafAt(newLeafCoord)->insertElement(subCoord, value);
        });
    });

    // g2p
    new_grid.foreachLeaf([&] (auto *leaf, Coord const &leafCoord) {
        leaf->foreachElement([&] (auto &value, int index) {
            Coord subCoord = leaf->indexToCoord(index);
            vec3f fpos = (leafCoord + subCoord * (1.f / (1 << L))) * grid.leaf_size;
            new_pos[value] = fpos;
        });
    });

    for (int i = 0; i < N; i++) {
        printf("%f %f\n", new_pos[i][0] - pos[i][0], vel[i][0] * dt);
        printf("%f %f\n", new_pos[i][1] - pos[i][1], vel[i][1] * dt);
        printf("%f %f\n", new_pos[i][2] - pos[i][2], vel[i][2] * dt);
    }
}
//for (int dz = 0; dz < 2; dz++) for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++) {
