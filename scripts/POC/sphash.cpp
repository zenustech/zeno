#include <glm/vec3.hpp>
#include <glm/glm.hpp>
#include <vector>
#include <cstdio>


// calc 60-bit morton code from 20-bit X,Y,Z fixed pos grid
static unsigned long morton3d(glm::vec3 const &pos) {
    auto v = glm::clamp(glm::vec<3, unsigned long>(
                glm::floor(pos * 1048576.0f)), 0ul, 1048575ul);
    static_assert(sizeof(v[0]) == 8);

    v = (v * 0x0000000100000001ul) & 0xFFFF00000000FFFFul;
    v = (v * 0x0000000000010001ul) & 0x00FF0000FF0000FFul;
    v = (v * 0x0000000000000101ul) & 0xF00F00F00F00F00Ful;
    v = (v * 0x0000000000000011ul) & 0x30C30C30C30C30C3ul;
    v = (v * 0x0000000000000005ul) & 0x4924924949249249ul;

    return v[0] * 4 + v[1] * 2 + v[2];
}

#define K 128
#define M 512

struct Entry {
    std::vector<int> pid;
} table[M];

static int hash(glm::vec3 const &pos) {
    auto v = glm::clamp(glm::ivec3(glm::floor(pos * (float)K)), 0, K);
    return (v[0] * 985211 + v[1] * 54321 + v[2] * 3141592 + 142857) % M;
}

std::vector<glm::vec3> pos;

int main(void) {
    pos.emplace_back(0.1, 0.1, 0);
    pos.emplace_back(0.1 + .1 / K, 0.1, 0);
    pos.emplace_back(0.1 + .2 / K, 0.1, 0);

    for (int i = 0; i < pos.size(); i++) {
        auto m = hash(pos[i]);
        table[m].pid.push_back(i);
    }

    auto m = hash(pos[0]);
    for (int j: table[m].pid) {
        printf("%f %f %f\n", pos[j].x, pos[j].y, pos[j].z);
    }
}
