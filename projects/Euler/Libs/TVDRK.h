#ifndef TVDRK_H
#define TVDRK_H

#include <Eigen/Eigen>
#include "Types.h"

namespace ZenEulerGas {
namespace Math {
namespace TimeIntegration {
/**
   Total variation diminishing Runge-Kutta method
*/

template <class DerivedV, class DerivedI>
Vector<DerivedV, 3> TVDRK2(const DerivedI& substep)
{
    BOW_ASSERT((substep == 0 || substep == 1));
    using T = DerivedV;
    Vector<T, 3> res;
    if (substep == 1) {
        res << (T).5, (T).5, (T).5;
    }
    else {
        res << (T)1, (T)0, (T)1;
    }
    return res;
};

template <class DerivedV, class DerivedI>
Vector<DerivedV, 3> TVDRK3(const DerivedI& substep)
{
    assertm((substep == 0 || substep == 1 || substep == 2), "TVDRK3 only takes substep 0,1,2") using T = DerivedV;
    Vector<T, 3> res;
    if (substep == 1) {
        res << T(.75), T(.25), T(.25);
    }
    else if (substep == 2) {
        res << (T)1 / (T)3, (T)2 / (T)3, (T)2 / (T)3;
    }
    else {
        res << (T)1, (T)0, (T)1;
    }
    return res;
};

}

}
} // namespace ZenEulerGas::Math::TimeIntegration

#endif
