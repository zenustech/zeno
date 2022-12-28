#include "../PBF/SPHKernels.h"
#include "../Utils/myPrint.h"
#include <vector>
#include <zeno/utils/vec.h>

float CubicKernel::m_radius;
float CubicKernel::m_k;
float CubicKernel::m_l;
float CubicKernel::m_W_zero;

int main()
{
    constexpr float h = 0.1;
    constexpr int num = 1000;
    CubicKernel::set(h);

    std::vector<float> cubicW(num);
    std::vector<zeno::vec3f> cubic_gradW(num);
    std::vector<float> poly6W(num);
    std::vector<zeno::vec3f> Spiky_gradW(num);
    for (size_t i = 0; i < num; i++)
    {
        //cubic W
        float dist = h/num * i;
        cubicW[i] = CubicKernel::W(dist);
        
        //cubic gradW
        zeno::vec3f distVec = zeno::vec3f{h/num*i,h/num*i,h/num*i};
        cubic_gradW[i] = CubicKernel::gradW(distVec);

        //poly6 W
        poly6W[i] = Poly6Kernel::W(dist);
        //spiky gradW
        Spiky_gradW[i] = SpikyKernel::gradW(distVec);
    }
    printScalarField("cubicW.csv",cubicW);
    printVectorField("cubic_gradW.csv",cubic_gradW);
    printScalarField("poly6W.csv",poly6W);
    printVectorField("Spiky_gradW.csv",Spiky_gradW);

}