#pragma once

#include "collision_utils.hpp"

namespace zeno {
namespace EDGE_EDGE_COLLISION {
    using namespace COLLISION_UTILS;

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
REAL psi(const std::vector<VECTOR3>& v,
                    const VECTOR2& a, 
                    const VECTOR2& b,
                    const REAL& _mu,
                    const REAL& _nu,
                    const REAL& _eps,
                    const REAL& _tooSmall)
{
    // convert to vertices and edges
    std::vector<VECTOR3> e{3};
    e[0] = v[3] - v[2];
    e[1] = v[0] - v[2];
    e[2] = v[1] - v[2];

    // get the interpolated vertices
    const VECTOR3 va = (a[0] * v[0] + a[1] * v[1]);
    const VECTOR3 vb = (b[0] * v[2] + b[1] * v[3]);
    if ((vb - va).norm() < _tooSmall)
        return 0.0;

    // there is not sign switch operation
    const REAL springLength = _eps - (vb - va).norm();
    return _mu * springLength * springLength;
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
REAL psiNegated(const std::vector<VECTOR3>& v,
                    const VECTOR2& a, 
                    const VECTOR2& b,
                    const REAL& _mu,
                    const REAL& _nu,
                    const REAL& _eps,
                    const REAL& _tooSmall)
{
    // convert to vertices and edges
    std::vector<VECTOR3> e{3};
    e[0] = v[3] - v[2];
    e[1] = v[0] - v[2];
    e[2] = v[1] - v[2];

    // get the interpolated vertices
    const VECTOR3 va = (a[0] * v[0] + a[1] * v[1]);
    const VECTOR3 vb = (b[0] * v[2] + b[1] * v[3]);
    if ((vb - va).norm() < _tooSmall)
        return 0.0;

    const REAL springLength = _eps + (vb - va).norm();
    return _mu * springLength * springLength;
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
VECTOR12 gradient(const std::vector<VECTOR3>& v,
                    const VECTOR2& a, 
                    const VECTOR2& b,
                    const REAL& _mu,
                    const REAL& _nu,
                    const REAL& _eps,
                    const REAL& _tooSmall)
{
    // convert to vertices and edges
    std::vector<VECTOR3> e{3};
    e[0] = v[3] - v[2];
    e[1] = v[0] - v[2];
    e[2] = v[1] - v[2];

    // get the interpolated vertices
    const VECTOR3 va = (a[0] * v[0] + a[1] * v[1]);
    const VECTOR3 vb = (b[0] * v[2] + b[1] * v[3]);
    const VECTOR3 diff = vb - va;

    // if the two are co-linear, give up
    // should probably fall back to cross-product formula here
    // (see EDGE_HYBRID_COLLISION)
    if (diff.norm() < _tooSmall)
        return VECTOR12::zeros();

    // get the normal
    VECTOR3 n = diff;
    n = n / n.norm();

    const REAL springLength = _eps - diff.norm();
    return (REAL)-2.0 * _mu * springLength * (vDiffPartial(a,b).transpose() * n);
}


///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
VECTOR12 gradientNegated(const std::vector<VECTOR3>& v,
                    const VECTOR2& a, 
                    const VECTOR2& b,
                    const REAL& _mu,
                    const REAL& _nu,
                    const REAL& _eps,
                    const REAL& _tooSmall)
{
    // convert to vertices and edges
    std::vector<VECTOR3> e{3};
    e[0] = v[3] - v[2];
    e[1] = v[0] - v[2];
    e[2] = v[1] - v[2];


    // get the interpolated vertices
    const VECTOR3 va = (a[0] * v[0] + a[1] * v[1]);
    const VECTOR3 vb = (b[0] * v[2] + b[1] * v[3]);
    const VECTOR3 diff = vb - va;

    // if the two are co-linear, give up
    // should probably fall back to cross-product formula here
    // (see EDGE_HYBRID_COLLISION)
    if (diff.norm() < _tooSmall)
        return VECTOR12::zeros();

    // get the direction
    VECTOR3 d = diff;
    d = d / d.norm();

    const REAL springLength = _eps + diff.norm();
    const MATRIX3x12 vPartial = vDiffPartial(a,b);
    
    return (REAL)2.0 * _mu * springLength * (vPartial.transpose() * d);
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
MATRIX12 hessian(const std::vector<VECTOR3>& v,
                    const VECTOR2& a, 
                    const VECTOR2& b,
                    const REAL& _mu,
                    const REAL& _nu,
                    const REAL& _eps,
                    const REAL& _tooSmall)
{
    // convert to vertices and edges
    std::vector<VECTOR3> e{3};
    e[0] = v[3] - v[2];
    e[1] = v[0] - v[2];
    e[2] = v[1] - v[2];

    // get the interpolated vertices
    const VECTOR3 va = (a[0] * v[0] + a[1] * v[1]);
    const VECTOR3 vb = (b[0] * v[2] + b[1] * v[3]);
    const VECTOR3 diff = vb - va;
    const REAL diffNorm = diff.norm();

    // if the two are co-linear, give up
    // should probably fall back to cross-product formula here
    // (see EDGE_HYBRID_COLLISION)
    if (diffNorm < _tooSmall)
        return MATRIX12::zeros();

    // get the normal
    VECTOR3 d = diff;
    d = d / d.norm();

    const MATRIX3x12 vPartial = vDiffPartial(a,b);
    const REAL invNorm = (diffNorm >= 1e-8) ? 1.0 / diffNorm : 1.0;
    const REAL invNorm3 = invNorm * invNorm * invNorm;

    const VECTOR12 normPartial = -invNorm * (vPartial.transpose() * diff);
    const MATRIX3x12 dGrad = invNorm * vPartial -
                            invNorm3 * zs::dyadic_prod(diff,(vPartial.transpose() * diff));

    return (REAL)-2.0 * _mu * ((_eps - diffNorm) * (vPartial.transpose() * dGrad) +
                        zs::dyadic_prod(normPartial,vPartial.transpose() * d));
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
MATRIX12 hessianNegated(const std::vector<VECTOR3>& v,
                    const VECTOR2& a, 
                    const VECTOR2& b,
                    const REAL& _mu,
                    const REAL& _nu,
                    const REAL& _eps,
                    const REAL& _tooSmall)
{
    // convert to vertices and edges
    std::vector<VECTOR3> e{3};
    e[0] = v[3] - v[2];
    e[1] = v[0] - v[2];
    e[2] = v[1] - v[2];

    // get the interpolated vertices
    const VECTOR3 va = (a[0] * v[0] + a[1] * v[1]);
    const VECTOR3 vb = (b[0] * v[2] + b[1] * v[3]);
    const VECTOR3 diff = vb - va;
    const REAL diffNorm = diff.norm();
    const REAL diffNorm3 = diffNorm * diffNorm * diffNorm;

    // if the two are co-linear, give up
    // should probably fall back to cross-product formula here
    // (see EDGE_HYBRID_COLLISION)
    if (diffNorm < _tooSmall)
        return MATRIX12::zeros();

    // get the normal
    VECTOR3 n = diff;
    n = n / n.norm();

    const MATRIX3x12 vPartial = vDiffPartial(a,b);
    const VECTOR12 normPartial = ((REAL)-1.0 / diffNorm) * (vPartial.transpose() * diff);

    const MATRIX3x12 nGrad = ((REAL)1.0 / diffNorm) * vPartial -
                            ((REAL)1.0 / diffNorm3) * zs::dyadic_prod(diff, (vPartial.transpose() * diff));

    // this is the energetically consistent one
    return (REAL)2.0 * _mu * ((_eps + diffNorm) * (vPartial.transpose() * nGrad) -
                        zs::dyadic_prod(normPartial,vPartial.transpose() * n));
}



};
};