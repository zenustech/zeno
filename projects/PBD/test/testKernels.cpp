#include "../PBF/SPHKernels.h"
#include "../Utils/myPrint.h"
#include <vector>
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
    for (size_t i = 0; i < num; i++)
    {
        float dist = h/num * i;
        cubicW[i] = CubicKernel::W(dist);
    }
    printScalarField("cubicW.csv",cubicW);
}