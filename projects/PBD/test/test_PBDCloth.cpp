// #define CATCH_CONFIG_MAIN
// #include "Catch2.hpp"

#include <zeno/utils/vec.h>
#include <cstdlib>
#include <vector>
#include <iostream>
#include "../Utils/myPrint.h"
#include "../Utils/myRand.h"

using namespace zeno;

/**
 * @brief 利用对角距离法求解弯折约束。
 * 
 * @param quads 三角形对。其中下标2和3代表对角
 * @param invMass 质量倒数
 * @param bendingRestLen 对角距离原长
 * @param bendingCompliance 参数：柔度
 * @param pos 输出：位置
 */
void solveBendingDistanceConstraints(
    const std::vector<vec4i> &quads,
    const std::vector<float> &invMass,
    const std::vector<float> &bendingRestLen,
    const float bendingCompliance,
    const float dt,
    std::vector<vec3f> &pos)
{
    auto alpha = bendingCompliance / dt /dt;

    for (auto i = 0; i < quads.size(); i++) 
    {
        int id0 = quads[i][2];
        int id1 = quads[i][3];

        auto w0 = invMass[id0];
        auto w1 = invMass[id1];
        auto w = w0 + w1;
        if (w == 0.0)
            continue;

        auto grads = pos[id0] - pos[id1];
        float Len = length(grads);
        if (Len == 0.0)
            continue;
        grads /= Len;
        auto C = Len - bendingRestLen[i];
        auto s = -C / (w + alpha);
        pos[id0] += grads *   s * invMass[id0];
        pos[id1] += grads * (-s * invMass[id1]);
    }
}


struct TestData
{
    unsigned numParticles=0;
    std::vector<vec3f> pos;
    std::vector<vec4i> quads;
    std::vector<float> invMass;
    std::vector<float> bendingRestLen;
    float bendingCompliance;
    float dt;

    TestData(unsigned numParticles_) : numParticles(numParticles_)
    {};

    void setNumParticles(unsigned numParticles_)
    {
        numParticles = numParticles_;
    }

    void gen()
    {
        std::cout<<"Generating the random data for the tested field...\n";
        std::cout<<"The number of particles is: "<<numParticles<<"\n";

        std::cout<<"Re-seeding...\n";
        reSeed();

        pos.resize(numParticles);
        fillRndVectorField(pos);
        std::cout<<"The input pos is: \n";
        printVectorFieldToScreen(pos);

        quads.resize(pos.size()/3);
        fillRndVectorField(quads);
        std::cout<<"The input quads is: \n";
        printVectorFieldToScreen(quads);

        invMass.resize(pos.size());
        fillRndScalarField(invMass);
        std::cout<<"The input invMass is: \n";
        printScalarFieldToScreen(invMass);

        bendingRestLen.resize(quads.size());
        fillRndScalarField(bendingRestLen);
        std::cout<<"The input bendingRestLen is: \n";
        printScalarFieldToScreen(bendingRestLen);

        bendingCompliance = genRnd();
        dt = genRnd();
        std::cout<<"The input bendingCompliance is:  "<<bendingCompliance<<std::endl;
        std::cout<<"The input dt is:  "<<dt<<std::endl;
    }
};


int main()
{
    TestData data(100);
    // data.setNumParticles(100);
    data.gen();

    solveBendingDistanceConstraints(data.quads, data.invMass, data.bendingRestLen, data.bendingCompliance, data.dt, data.pos);

    std::cout<<"The result pos is: \n";
    printVectorFieldToScreen(data.pos);

    return 0;
}

// TEST_CASE("test_bendingConstaint", "test1")
// {
//     cout << rand() << endl;
//     std::vector<vec4i> quads{1,2,3,4};
//     std::vector<float> invMass{0.1,0.1,0.2,0.1};
//     solveBendingDistanceConstraints()
// }