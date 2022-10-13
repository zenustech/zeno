#pragma once

#include "Structures.hpp"
#include "Utils.hpp"

#include "zensim/math/VecInterface.hpp"

namespace zeno {
namespace EDGE_EDGE_COLLISION {

    using REAL = float;
    using VECTOR12 = typename zs::vec<REAL,12>;
    using VECTOR3 = zs::vec<REAL,3>;
    using MATRIX3x12 = typename zs::vec<REAL,3,12>;
    using MATRIX12 = typename zs::vec<REAL,12,12>;

    EDGE_COLLISION::EDGE_COLLISION(const REAL& mu, const REAL& eps) :
        _mu(mu), _eps(eps)
    {
    }

    std::string EDGE_COLLISION::name() const
    {
    return "EDGE_COLLISION";
    }

    ///////////////////////////////////////////////////////////////////////
    // convert the 12-vector in a way that imposes a consistent tet 
    // ordering for vertices and edges
    ///////////////////////////////////////////////////////////////////////
    void EDGE_COLLISION::getVerticesAndEdges(const VECTOR12& x,
                                            vector<VECTOR3>& v,
                                            vector<VECTOR3>& e)
    {
    v.resize(4);
    for (int i = 0; i < 4; i++)
    {
        v[i][0] = x[i * 3];
        v[i][1] = x[i * 3 + 1];
        v[i][2] = x[i * 3 + 2];
    }

    e.resize(2);
    e[0] = v[1] - v[0];
    e[1] = v[3] - v[2];
    }

    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    REAL EDGE_COLLISION::psi(const vector<VECTOR3>& vertices,
                            const VECTOR2& a, 
                            const VECTOR2& b) const
    {
    return psi(flattenVertices(vertices),a,b);
    }

    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    REAL EDGE_COLLISION::psiNegated(const vector<VECTOR3>& vertices,
                                    const VECTOR2& a, 
                                    const VECTOR2& b) const
    {
    return psiNegated(flattenVertices(vertices),a,b);
    }

    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    VECTOR12 EDGE_COLLISION::gradient(const vector<VECTOR3>& vertices,
                                    const VECTOR2& a, 
                                    const VECTOR2& b) const
    {
    return gradient(flattenVertices(vertices),a,b);
    }

    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    VECTOR12 EDGE_COLLISION::gradientNegated(const vector<VECTOR3>& vertices,
                                            const VECTOR2& a, 
                                            const VECTOR2& b) const
    {
    return gradientNegated(flattenVertices(vertices),a,b);
    }

    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    MATRIX12 EDGE_COLLISION::hessian(const vector<VECTOR3>& vertices,
                                    const VECTOR2& a, 
                                    const VECTOR2& b) const
    {
    return hessian(flattenVertices(vertices),a,b);
    }

    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    MATRIX12 EDGE_COLLISION::hessianNegated(const vector<VECTOR3>& vertices,
                                            const VECTOR2& a, 
                                            const VECTOR2& b) const
    {
    return hessianNegated(flattenVertices(vertices),a,b);
    }

    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    REAL EDGE_COLLISION::psi(const VECTOR12& x,
                            const VECTOR2& a, 
                            const VECTOR2& b) const
    {
    // convert to vertices and edges
    vector<VECTOR3> v;
    vector<VECTOR3> e;
    getVerticesAndEdges(x,v,e);

    // get the normal
    VECTOR3 n = e[1].cross(e[0]);

    // if the two are co-linear, skip the normalization 
    if (n.norm() > 1e-8)
        n = n / n.norm();

    // get the interpolated vertices
    const VECTOR3 va = (a[0] * v[0] + a[1] * v[1]);
    const VECTOR3 vb = (b[0] * v[2] + b[1] * v[3]);
    const VECTOR3 diff = vb - va;
    
    const REAL sign = (diff.dot(n) > 0.0) ? -1.0 : 1.0;

    // get the spring length, non-zero rest-length
    const REAL springLength = _eps + diff.dot(sign * n);
    return _mu * springLength * springLength;
    }

    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    REAL EDGE_COLLISION::psiNegated(const VECTOR12& x,
                                    const VECTOR2& a, 
                                    const VECTOR2& b) const
    {
    // convert to vertices and edges
    vector<VECTOR3> v;
    vector<VECTOR3> e;
    getVerticesAndEdges(x,v,e);

    // Harmon  et al. says that if the two edges are nearly parallel, a vertex-face
    // will pick up the slack, so ignore it. But ... the collision is still well
    // defined from the perspective of the pinned version of this energy.
    //
    // Regardless, the conditioning of the force goes crazy in the near-parallel case,
    // so we should skip it here.
    if (nearlyParallel(e))
        return 0.0;

    // get the normal
    VECTOR3 n = e[1].cross(e[0]);

    // if the two are co-linear, skip the normalization (this may be redundant with
    // the nearlyParallel check)
    if (n.norm() > 1e-8)
        n = n / n.norm();

    // get the interpolated vertices
    const VECTOR3 va = (a[0] * v[0] + a[1] * v[1]);
    const VECTOR3 vb = (b[0] * v[2] + b[1] * v[3]);
    const VECTOR3 diff = vb - va;
    
    const REAL sign = (diff.dot(n) > 0.0) ? -1.0 : 1.0;

    // get the spring length, non-zero rest-length
    const REAL springLength = _eps - diff.dot(sign * n);
    return _mu * springLength * springLength;
    }

    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    VECTOR12 EDGE_COLLISION::gradient(const VECTOR12& x, 
                                    const VECTOR2& a, 
                                    const VECTOR2& b) const
    {
    // convert to vertices and edges
    vector<VECTOR3> v;
    vector<VECTOR3> e;
    getVerticesAndEdges(x,v,e);

    assert(v.size() == 4);
    assert(e.size() == 2);

    // Harmon  et al. says that if the two edges are nearly parallel, a vertex-face
    // will pick up the slack, so ignore it. But ... the collision is still well
    // defined from the perspective of the pinned version of this energy.
    //
    // Regardless, the conditioning of the force goes crazy in the near-parallel case,
    // so we should skip it here.
    if (nearlyParallel(e))
        return VECTOR12::Zero();

    // get the normal
    VECTOR3 n = e[1].cross(e[0]);
    
    // if the two are co-linear, skip the normalization 
    if (n.norm() > 1e-8)
        n = n / n.norm();
    
    // get the interpolated vertices
    const VECTOR3 va = (a[0] * v[0] + a[1] * v[1]);
    const VECTOR3 vb = (b[0] * v[2] + b[1] * v[3]);
    const VECTOR3 diff = vb - va;

    const REAL sign = (diff.dot(n) > 0.0) ? -1.0 : 1.0;
    const REAL springLength = _eps + diff.dot(sign * n);
    return 2.0 * _mu * springLength * springLengthGradient(e,n,diff,a,b);
    }

    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    VECTOR12 EDGE_COLLISION::gradientNegated(const VECTOR12& x, 
                                            const VECTOR2& a, 
                                            const VECTOR2& b) const
    {
    // convert to vertices and edges
    vector<VECTOR3> v;
    vector<VECTOR3> e;
    getVerticesAndEdges(x,v,e);

    assert(v.size() == 4);
    assert(e.size() == 2);

    // Harmon  et al. says that if the two edges are nearly parallel, a vertex-face
    // will pick up the slack, so ignore it. But ... the collision is still well
    // defined from the perspective of the pinned version of this energy.
    //
    // Regardless, the conditioning of the force goes crazy in the near-parallel case,
    // so we should skip it here.
    if (nearlyParallel(e))
        return VECTOR12::Zero();

    // get the normal
    VECTOR3 n = e[1].cross(e[0]);
    
    // if the two are co-linear, skip the normalization (this may be redundant with
    // the nearlyParallel check)
    if (n.norm() > 1e-8)
        n = n / n.norm();
    
    // get the interpolated vertices
    const VECTOR3 va = (a[0] * v[0] + a[1] * v[1]);
    const VECTOR3 vb = (b[0] * v[2] + b[1] * v[3]);
    const VECTOR3 diff = vb - va;

    const REAL sign = (diff.dot(n) > 0.0) ? -1.0 : 1.0;

    //const REAL springLength = _eps + diff.dot(sign * n);
    //return 2.0 * _mu * springLength * springLengthGradient(e,n,diff,a,b);
    const REAL springLength = _eps - diff.dot(sign * n);
    return -2.0 * _mu * springLength * springLengthGradient(e,n,diff,a,b);
    }

    ///////////////////////////////////////////////////////////////////////
    // partial of (va - vb)
    ///////////////////////////////////////////////////////////////////////
    MATRIX3x12 EDGE_COLLISION::vDiffPartial(const VECTOR2& a, const VECTOR2& b)
    {
    MATRIX3x12 tPartial;
    tPartial.setZero();
    tPartial(0,0) = tPartial(1,1)  = tPartial(2,2) = -a[0];
    tPartial(0,3) = tPartial(1,4)  = tPartial(2,5) = -a[1];
    tPartial(0,6) = tPartial(1,7)  = tPartial(2,8) = b[0];
    tPartial(0,9) = tPartial(1,10) = tPartial(2,11) = b[1];

    return tPartial;
    }

    ///////////////////////////////////////////////////////////////////////
    // gradient of spring length, n' * (va - vb)
    ///////////////////////////////////////////////////////////////////////
    VECTOR12 EDGE_COLLISION::springLengthGradient(const std::vector<VECTOR3>& e,
                                                const VECTOR3& n,
                                                const VECTOR3& diff,
                                                const VECTOR2& a,
                                                const VECTOR2& b)
    {
    MATRIX3x12 nPartial = normalGradientEE(e);
    MATRIX3x12 tPartial = vDiffPartial(a,b);
    const REAL sign = (diff.dot(n) > 0.0) ? -1.0 : 1.0;
    return sign * nPartial.transpose() * diff + tPartial.transpose() * (sign * n);
    }

    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    MATRIX12 EDGE_COLLISION::hessian(const VECTOR12& x,
                                    const VECTOR2& a, 
                                    const VECTOR2& b) const
    {
    // convert to vertices and edges
    vector<VECTOR3> v;
    vector<VECTOR3> e;
    getVerticesAndEdges(x, v, e);
    assert(v.size() == 4);
    assert(e.size() == 2);
    
    // Harmon  et al. says that if the two edges are nearly parallel, a vertex-face
    // will pick up the slack, so ignore it. But ... the collision is still well
    // defined from the perspective of the pinned version of this energy.
    //
    // Regardless, the conditioning of the force goes crazy in the near-parallel case,
    // so we should skip it here.
    if (nearlyParallel(e))
        return MATRIX12::Zero();

    // get the normal
    VECTOR3 n = e[1].cross(e[0]);
    
    // if the two are co-linear, skip the normalization 
    if (n.norm() > 1e-8)
        n = n / n.norm();

    // get the interpolated vertices
    const VECTOR3 va = (a[0] * v[0] + a[1] * v[1]);
    const VECTOR3 vb = (b[0] * v[2] + b[1] * v[3]);
    const VECTOR3 diff = vb - va;

    //if (diff.dot(n) > 0.0)
    //  n *= -1.0;
    const REAL sign = (diff.dot(n) > 0.0) ? -1.0 : 1.0;

    // get the spring length, non-zero rest-length
    const REAL springLength = _eps + diff.dot(sign * n);

    // ndotGrad    = ndot_gradient(x);
    const VECTOR12 springLengthGrad = springLengthGradient(e,n,diff,a,b);

    // ndotHessian = ndot_hessian(x);
    const MATRIX12 springLengthH = springLengthHessian(e,n,diff,a,b);
    
    return 2.0 * _mu * (springLengthGrad * springLengthGrad.transpose() +
                        springLength * springLengthH);
    }

    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    MATRIX12 EDGE_COLLISION::hessianNegated(const VECTOR12& x,
                                            const VECTOR2& a, 
                                            const VECTOR2& b) const
    {
    // convert to vertices and edges
    vector<VECTOR3> v;
    vector<VECTOR3> e;
    getVerticesAndEdges(x, v, e);
    assert(v.size() == 4);
    assert(e.size() == 2);
    
    // Harmon  et al. says that if the two edges are nearly parallel, a vertex-face
    // will pick up the slack, so ignore it. But ... the collision is still well
    // defined from the perspective of the pinned version of this energy.
    //
    // Regardless, the conditioning of the force goes crazy in the near-parallel case,
    // so we should skip it here.
    if (nearlyParallel(e))
        return MATRIX12::Zero();

    // get the normal
    VECTOR3 n = e[1].cross(e[0]);
    
    // if the two are co-linear, skip the normalization (this may be redundant with
    // the nearlyParallel check)
    if (n.norm() > 1e-8)
        n = n / n.norm();

    // get the interpolated vertices
    const VECTOR3 va = (a[0] * v[0] + a[1] * v[1]);
    const VECTOR3 vb = (b[0] * v[2] + b[1] * v[3]);
    const VECTOR3 diff = vb - va;

    //if (diff.dot(n) > 0.0)
    //  n *= -1.0;
    const REAL sign = (diff.dot(n) > 0.0) ? -1.0 : 1.0;

    // get the spring length, non-zero rest-length
    //const REAL springLength = _eps + diff.dot(sign * n);
    const REAL springLength = _eps - diff.dot(sign * n);

    // ndotGrad    = ndot_gradient(x);
    const VECTOR12 springLengthGrad = springLengthGradient(e,n,diff,a,b);

    // ndotHessian = ndot_hessian(x);
    const MATRIX12 springLengthH = springLengthHessian(e,n,diff,a,b);
    
    //return 2.0 * _mu * (springLengthGrad * springLengthGrad.transpose() +
    //                    springLength * springLengthH);
    return -2.0 * _mu * (springLength * springLengthH - 
                        springLengthGrad * springLengthGrad.transpose());
    }

    ///////////////////////////////////////////////////////////////////////
    // hessian of spring length, n' * (v[2] - v[0])
    ///////////////////////////////////////////////////////////////////////
    MATRIX12 EDGE_COLLISION::springLengthHessian(const std::vector<VECTOR3>& e,
                                                const VECTOR3& n,
                                                const VECTOR3& diff,
                                                const VECTOR2& a,
                                                const VECTOR2& b)
    {
    MATRIX3x12 tPartial = vDiffPartial(a,b);
    const REAL sign = (diff.dot(n) > 0.0) ? -1.0 : 1.0;

    //% mode-3 contraction
    //[nx ny nz] = normal_hessian(x);
    //final = nx * delta(1) + ny * delta(2) + nz * delta(3);
    vector<MATRIX12> normalH = normalHessianEE(e);

    MATRIX12 contracted = diff[0] * normalH[0] + 
                            diff[1] * normalH[1] + 
                            diff[2] * normalH[2];
    contracted *= sign;
    
    //nGrad= normal_gradient(x);
    MATRIX3x12 nGrad = sign * normalGradientEE(e);

    //product = nGrad' * vGrad;
    //final = final + product + product';
    MATRIX12 product = nGrad.transpose() * tPartial;

    return contracted + product + product.transpose();
    }

    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    MATRIX12 EDGE_COLLISION::clampedHessian(const VECTOR12& x,
                                            const VECTOR2& a, 
                                            const VECTOR2& b) const
    {
    return clampEigenvalues(hessian(x,a,b));
    }

    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    MATRIX12 EDGE_COLLISION::clampedHessianNegated(const VECTOR12& x,
                                                const VECTOR2& a, 
                                                const VECTOR2& b) const
    {
    return clampEigenvalues(hessianNegated(x,a,b));
    }

    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    MATRIX12 EDGE_COLLISION::clampedHessian(const vector<VECTOR3>& v,
                                            const VECTOR2& a, 
                                            const VECTOR2& b) const
    {
    return clampedHessian(flattenVertices(v),a,b);
    }

    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    MATRIX12 EDGE_COLLISION::clampedHessianNegated(const vector<VECTOR3>& v,
                                                const VECTOR2& a, 
                                                const VECTOR2& b) const
    {
    return clampedHessianNegated(flattenVertices(v),a,b);
    }

    ///////////////////////////////////////////////////////////////////////
    // are the two edges nearly parallel?
    ///////////////////////////////////////////////////////////////////////
    bool EDGE_COLLISION::nearlyParallel(const vector<VECTOR3> e)
    {
    const VECTOR3 e0 = e[0].normalized();
    const VECTOR3 e1 = e[1].normalized();
    const REAL dotted = fabs(e0.dot(e1));

    // too conservative, still seeing some conditioning problems
    // in the simulation. If the mesh suddenly pops, it means
    // that the conditioning problem made the solve go haywire.
    //const REAL eps = 1e-4;
    
    // this is still quite conservative, with some popping visible
    // in the simulation
    //const REAL eps = 1e-3;
    
    // this seems too permissive, and ends up missing some collisions,
    // but is what we're using for now
    const REAL eps = 1e-2;

    return (dotted > 1.0 - eps);
    }


};
};