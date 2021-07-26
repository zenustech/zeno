#include <glm/vec3.hpp>
#include <glm/glm.hpp>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cmath>
#include <array>
#include "Grid.h"
#include "HashGrid.h"
#include "LeafSpecs.h"

int main() {
    Grid<HashGrid<Points<int, 32>>> grid, new_grid;

    const int L = grid.LeafShift;
    const int N = 3;
    std::vector<glm::vec3> pos(N);
    std::vector<glm::vec3> new_pos(N);
    std::vector<glm::vec3> vel(N);
    for (int i = 0; i < N; i++) {
        pos[i] = (glm::vec3)glm::ivec3(
                rand(), rand(), rand()) / (float)RAND_MAX;
        vel[i] = (glm::vec3)glm::ivec3(
                rand(), rand(), rand()) / (float)RAND_MAX * 2.f - 1.f;
    }

    float leaf_size = 0.04f;
    float dt = 0.01f;

    // p2g
    for (int i = 0; i < N; i++) {
        auto iipos = pos[i] / leaf_size;
        auto ipos = glm::ivec3(iipos);
        auto jpos = glm::ivec3(glm::mod(iipos, 1.f) * float(1 << L));
        Coord leafCoord{ipos.x, ipos.y, ipos.z};
        auto *leaf = grid.leafAt(leafCoord);
        Coord subCoord{jpos.x, jpos.y, jpos.z};
        leaf->insertElement(subCoord, i);
        //printf("? %d %d %d\n", leafCoord.x, leafCoord.y, leafCoord.z);
        //printf("! %d %d %d\n", subCoord.x, subCoord.y, subCoord.z);
    }

    // advect
    grid.foreachLeaf([&] (auto *leaf, Coord const &leafCoord) {
        leaf->foreachElement([&] (auto &value, int index) {
            Coord subCoord = leaf->indexToCoord(index);
            auto vel_dt = vel[value] * dt;
            subCoord.x += int(vel_dt.x * float(1 << L) / leaf_size);
            subCoord.y += int(vel_dt.y * float(1 << L) / leaf_size);
            subCoord.z += int(vel_dt.z * float(1 << L) / leaf_size);

            Coord newLeafCoord = leafCoord;
            newLeafCoord.x += subCoord.x >> L;
            newLeafCoord.y += subCoord.y >> L;
            newLeafCoord.z += subCoord.z >> L;

            subCoord.x &= (1 << L) - 1;
            subCoord.y &= (1 << L) - 1;
            subCoord.z &= (1 << L) - 1;

            //printf("? %d %d %d\n", newLeafCoord.x, newLeafCoord.y, newLeafCoord.z);
            //printf("! %d %d %d\n", subCoord.x, subCoord.y, subCoord.z);
            new_grid.leafAt(newLeafCoord)->insertElement(subCoord, value);
        });
    });

    // g2p
    new_grid.foreachLeaf([&] (auto *leaf, Coord const &leafCoord) {
        //printf("? %d %d %d\n", leafCoord.x, leafCoord.y, leafCoord.z);
        leaf->foreachElement([&] (auto &value, int index) {
            Coord subCoord = leaf->indexToCoord(index);
            //printf("! %d %d %d\n", subCoord.x, subCoord.y, subCoord.z);
            float fx = (leafCoord.x + subCoord.x / float(1 << L)) * leaf_size;
            float fy = (leafCoord.y + subCoord.y / float(1 << L)) * leaf_size;
            float fz = (leafCoord.z + subCoord.z / float(1 << L)) * leaf_size;

            new_pos[value] = glm::vec3(fx, fy, fz);
        });
    });

    for (int i = 0; i < N; i++) {
        printf("%f %f\n", new_pos[i].x - pos[i].x, vel[i].x * dt);
        printf("%f %f\n", new_pos[i].y - pos[i].y, vel[i].y * dt);
        printf("%f %f\n", new_pos[i].z - pos[i].z, vel[i].z * dt);
    }
}
//for (int dz = 0; dz < 2; dz++) for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++) {
