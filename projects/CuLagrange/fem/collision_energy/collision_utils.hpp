#pragma once

#include "Structures.hpp"
#include "Utils.hpp"

#include "zensim/math/VecInterface.hpp"


namespace zeno {
namespace COLLISION_UTILS {

    // using namespace std;

    using REAL = float;
    using VECTOR12 = typename zs::vec<REAL,12>;
    using VECTOR4 = typename zs::vec<REAL,4>;
    using VECTOR3 = typename zs::vec<REAL,3>;
    using VECTOR2 = typename zs::vec<REAL,2>;
    using MATRIX3x12 = typename zs::vec<REAL,3,12>;
    using MATRIX12 = typename zs::vec<REAL,12,12>;

    ///////////////////////////////////////////////////////////////////////
    // should we reverse the direction of the force?
    ///////////////////////////////////////////////////////////////////////
    constexpr bool reverse(const VECTOR3 v[3],const VECTOR3 e[3])
    {
        // get the normal
        VECTOR3 n = e[2].cross(e[0]);
        n = n / n.norm();

        // e[1] is already the collision vertex recentered to the origin
        // (v[0] - v[2])
        const REAL dotted = n.dot(e[1]);
        return (dotted < 0) ? true : false;
    }


    constexpr VECTOR12 flatten(const VECTOR3 v[4]) {
        auto res = VECTOR12::zeros();
        for(size_t i = 0;i < 4;++i)
            for(size_t j = 0;j < 3;++j)
                res[i * 3 + j] = v[i][j];
        return res;
    }

    constexpr void setCol(MATRIX3x12& m,int col,const VECTOR3& v) {
        for(int i = 0;i < 3;++i)
            m[i][col] = v[i];
    }

    constexpr VECTOR3 getCol(const MATRIX3x12& m,int col) {
        VECTOR3 res{0};
        for(int i = 0;i < 3;++i)
            res[i] = m[i][col];
        return res;
    }


    ///////////////////////////////////////////////////////////////////////
    // partial of (va - vb)
    ///////////////////////////////////////////////////////////////////////
    constexpr MATRIX3x12 vDiffPartial(const VECTOR2& a, const VECTOR2& b)
    {
        auto tPartial = MATRIX3x12::zeros();
        tPartial(0,0) = tPartial(1,1)  = tPartial(2,2) = -a[0];
        tPartial(0,3) = tPartial(1,4)  = tPartial(2,5) = -a[1];
        tPartial(0,6) = tPartial(1,7)  = tPartial(2,8) = b[0];
        tPartial(0,9) = tPartial(1,10) = tPartial(2,11) = b[1];

        return tPartial;
    }

 ///////////////////////////////////////////////////////////////////////
    // gradient of the cross product used to compute the normal,
    // edge-edge case
    ///////////////////////////////////////////////////////////////////////
    constexpr MATRIX3x12 crossGradientEE(const VECTOR3 e[3])
    {
        MATRIX3x12 crossMatrix{};

        const REAL e0x = e[0][0];
        const REAL e0y = e[0][1];
        const REAL e0z = e[0][2];

        const REAL e1x = e[1][0];
        const REAL e1y = e[1][1];
        const REAL e1z = e[1][2];

        setCol(crossMatrix,0,VECTOR3(0, -e1z, e1y));
        setCol(crossMatrix,1,VECTOR3(e1z, 0, -e1x));
        setCol(crossMatrix,2,VECTOR3(-e1y, e1x, 0));

        setCol(crossMatrix,3,VECTOR3(0, e1z, -e1y));
        setCol(crossMatrix,4,VECTOR3(-e1z, 0, e1x));
        setCol(crossMatrix,5,VECTOR3(e1y, -e1x, 0));

        setCol(crossMatrix,6,VECTOR3(0, e0z, -e0y));
        setCol(crossMatrix,7,VECTOR3(-e0z, 0, e0x));
        setCol(crossMatrix,8,VECTOR3(e0y, -e0x, 0));

        setCol(crossMatrix,9,VECTOR3(0, -e0z, e0y));
        setCol(crossMatrix,10,VECTOR3(e0z, 0, -e0x));
        setCol(crossMatrix,11,VECTOR3(-e0y, e0x, 0));

        return crossMatrix;
    }


    ///////////////////////////////////////////////////////////////////////
    // gradient of the normal, edge-edge case
    ///////////////////////////////////////////////////////////////////////
    constexpr MATRIX3x12 normalGradientEE(const VECTOR3 e[3])
    {
        VECTOR3 crossed = e[1].cross(e[0]);
        const REAL crossNorm = crossed.norm();
        const REAL crossNormInv = (crossNorm > 1e-8) ? 1.0 / crossed.norm() : 0.0;
        const REAL crossNormCubedInv = (crossNorm > 1e-8) ? 1.0 / zs::pow(crossed.dot(crossed), (REAL)1.5) : 0.0;
        MATRIX3x12 crossMatrix = crossGradientEE(e);

        MATRIX3x12 result{};
        for (int i = 0; i < 12; i++)
        {
            VECTOR3 crossColumn = getCol(crossMatrix,i);
            auto col_vec = crossNormInv * crossColumn - 
                            ((crossed.dot(crossColumn)) * crossNormCubedInv) * crossed;
            setCol(result,i,col_vec);
        }
        return result;
    }

    ///////////////////////////////////////////////////////////////////////
    // one entry of the rank-3 hessian of the cross product used to compute 
    // the triangle normal, edge-edge case
    ///////////////////////////////////////////////////////////////////////
    constexpr VECTOR3 crossHessianEE(const int iIn, const int jIn)
    {
        int i = iIn;
        int j = jIn;

        if (i > j)
        {
            int temp = j;
            j = i;
            i = temp;
        }

        if ((i == 1 && j == 11)  || (i == 2 && j == 7) || (i == 4 && j == 8) || (i == 5 && j == 10))
            return VECTOR3(1, 0, 0);

        if ((i == 0 && j == 8) || (i == 2 && j == 9) || (i == 3 && j == 11) || (i == 5 && j == 6))
            return VECTOR3(0, 1, 0);

        if ((i == 0 && j == 10)  || (i == 1 && j == 6) || (i == 3 && j == 7) || (i == 4 && j == 9))
            return VECTOR3(0, 0, 1);

        if ((i == 1 && j == 8) || (i == 2 && j == 10) || (i == 4 && j == 11) || (i == 5 && j == 7))
            return VECTOR3(-1, 0, 0);

        if ((i == 0 && j == 11) || (i == 2 && j == 6) || (i == 3 && j == 8) || (i == 5 && j == 9))
            return VECTOR3(0, -1, 0);

        if ((i == 0 && j == 7) || (i == 1 && j == 9) || (i == 3 && j == 10) || (i == 4 && j == 6))
            return VECTOR3(0, 0, -1);

        return VECTOR3(0, 0, 0);
    }

    ///////////////////////////////////////////////////////////////////////
    // hessian of the triangle normal, edge-edge case
    ///////////////////////////////////////////////////////////////////////
    constexpr void normalHessianEE(const VECTOR3 e[3],MATRIX12 H[3])
    {
        // using namespace std;

        // MATRIX12 H[3];
        for (int i = 0; i < 3; i++)
            H[i] = MATRIX12::zeros();

        VECTOR3 crossed = e[1].cross(e[0]);
        MATRIX3x12 crossGrad = crossGradientEE(e);
        const VECTOR3& z = crossed;
        
        //denom15 = (z' * z) ^ (1.5);
        REAL denom15 = zs::pow(crossed.dot(crossed), (REAL)1.5);
        REAL denom25 = zs::pow(crossed.dot(crossed), (REAL)2.5);

        for (int j = 0; j < 12; j++)
            for (int i = 0; i < 12; i++)
            {
                VECTOR3 zGradi = getCol(crossGrad,i);
                VECTOR3 zGradj = getCol(crossGrad,j);
                VECTOR3 zHessianij = crossHessianEE(i,j);

                // z = cross(e2, e0);
                // zGrad = crossGradientVF(:,i);
                // alpha= (z' * zGrad) / (z' * z) ^ (1.5);
                REAL a = z.dot(zGradi) / denom15;

                // final = (zGradj' * zGradi) / denom15 + (z' * cross_hessian(i,j)) / denom15;
                // final = final - 3 * ((z' * zGradi) / denom25) * (zGradj' * z);
                REAL aGrad = (zGradj.dot(zGradi)) / denom15 + 
                            z.dot(crossHessianEE(i,j)) / denom15;
                aGrad -= 3.0 * (z.dot(zGradi) / denom25) * zGradj.dot(z);
                
                //entry = -((zGradj' * z) / denom15) * zGradi + 
                //          1 / norm(z) * zHessianij - 
                //          alpha * zGradj - alphaGradj * z;
                VECTOR3 entry = -((zGradj.dot(z)) / denom15) * zGradi + 
                                    (REAL)1.0 / z.norm() * zHessianij - 
                                    a * zGradj - aGrad * z;

                H[0](i,j) = entry[0];
                H[1](i,j) = entry[1];
                H[2](i,j) = entry[2];
            }
        // return H;
    }



    constexpr MATRIX3x12 crossGradientVF(const VECTOR3 e[3])
    {
        MATRIX3x12 crossMatrix{};

        const REAL e0x = e[0][0];
        const REAL e0y = e[0][1];
        const REAL e0z = e[0][2];

        const REAL e2x = e[2][0];
        const REAL e2y = e[2][1];
        const REAL e2z = e[2][2];

        setCol(crossMatrix,0,VECTOR3(0,0,0)); 
        setCol(crossMatrix,1,VECTOR3(0,0,0)); 
        setCol(crossMatrix,2,VECTOR3(0,0,0)); 
        setCol(crossMatrix,3,VECTOR3(0, -e0z, e0y)); 
        setCol(crossMatrix,4,VECTOR3(e0z, 0, -e0x)); 
        setCol(crossMatrix,5,VECTOR3(-e0y, e0x, 0));
        setCol(crossMatrix,6,VECTOR3(0, (e0z - e2z), (-e0y + e2y)));
        setCol(crossMatrix,7,VECTOR3((-e0z + e2z), 0, (e0x - e2x)));
        setCol(crossMatrix,8,VECTOR3((e0y - e2y), (-e0x + e2x), 0));
        setCol(crossMatrix,9,VECTOR3(0, e2z, -e2y));
        setCol(crossMatrix,10,VECTOR3(-e2z, 0, e2x));
        setCol(crossMatrix,11,VECTOR3(e2y, -e2x, 0));

        return crossMatrix;
    }

    constexpr MATRIX3x12 normalGradientVF(const VECTOR3 e[3])
    {
        //crossed = cross(e2, e0);
        VECTOR3 crossed = e[2].cross(e[0]);
        REAL crossNorm = crossed.norm();
        const REAL crossNormCubedInv = 1.0 / zs::pow(crossed.dot(crossed), (REAL)1.5);
        MATRIX3x12 crossMatrix = crossGradientVF(e);

        //final = zeros(3,12);
        //for i = 1:12
        //  crossColumn = crossMatrix(:,i);
        //  final(:,i) = (1 / crossNorm) * crossColumn - ((crossed' * crossColumn) / crossNormCubed) * crossed;
        //end
        MATRIX3x12 result{};
        for (int i = 0; i < 12; i++)
        {
            auto crossColumn = VECTOR3(crossMatrix[0][i],
                crossMatrix[1][i],
                crossMatrix[2][i]);
            VECTOR3 resc = ((REAL)1. / crossNorm) * crossColumn - 
                            ((crossed.dot(crossColumn)) * crossNormCubedInv) * crossed;
            setCol(result,i,resc);
        }
        return result;
    }

    ///////////////////////////////////////////////////////////////////////
    // gradient of spring length, n' * (va - vb)
    ///////////////////////////////////////////////////////////////////////
    constexpr VECTOR12 springLengthGradient(const VECTOR3 e[3],
                                                const VECTOR3& n,
                                                const VECTOR3& diff,
                                                const VECTOR2& a,
                                                const VECTOR2& b){
        MATRIX3x12 nPartial = normalGradientEE(e);
        MATRIX3x12 tPartial = vDiffPartial(a,b);
        const REAL sign = (diff.dot(n) > (REAL)0.0) ? (REAL)-1.0 : (REAL)1.0;
        return sign * nPartial.transpose() * diff + tPartial.transpose() * (sign * n);
    }

    constexpr VECTOR12 springLengthGradient(const VECTOR3 v[3],const VECTOR3 e[3],const VECTOR3& n)
    {
        const MATRIX3x12 nPartial = normalGradientVF(e);
        const VECTOR3 tvf = v[0] - v[2];

        MATRIX3x12 tvfPartial{0};
        tvfPartial(0,0) = tvfPartial(1,1) = tvfPartial(2,2) = 1.0;
        tvfPartial(0,6) = tvfPartial(1,7) = tvfPartial(2,8) = -1.0;

        //f = nPartial' * (v2 - v0) + tvfPartial' * n;
        return nPartial.transpose() * tvf + tvfPartial.transpose() * n;
    }   
///////////////////////////////////////////////////////////////////////
// one entry of the rank-3 hessian of the cross product used to compute 
// the triangle normal, vertex-face case
///////////////////////////////////////////////////////////////////////
    constexpr VECTOR3 crossHessianVF(const int iIn, const int jIn)
    {
        int i = iIn;
        int j = jIn;

        if (i > j)
        {
            int temp = j;
            j = i;
            i = temp;
        }

        if ((i == 5 && j == 7)  || (i == 8 && j == 10) || (i == 4 && j == 11))
            return VECTOR3(1, 0, 0);

        if ((i == 6 && j == 11) || (i == 3 && j == 8) || (i == 5 && j == 9))
            return VECTOR3(0, 1, 0);

        if ((i == 4 && j == 6)  || (i == 7 && j == 9) || (i == 3 && j == 10))
            return VECTOR3(0, 0, 1);

        if ((i == 7 && j == 11) || (i == 4 && j == 8) || (i == 5 && j == 10))
            return VECTOR3(-1, 0, 0);

        if ((i == 5 && j == 6)  || (i == 8 && j == 9) || (i == 3 && j == 11))
            return VECTOR3(0, -1, 0);

        if ((i == 6 && j == 10) || (i == 3 && j == 7) || (i == 4 && j == 9))
            return VECTOR3(0, 0, -1);

        return VECTOR3(0, 0, 0);
    }

///////////////////////////////////////////////////////////////////////
// hessian of the triangle normal, vertex-face case
///////////////////////////////////////////////////////////////////////
    constexpr void normalHessianVF(const VECTOR3 e[3],MATRIX12 H[3])
    {
        // using namespace std;

        // MATRIX12 H[3];
        for (int i = 0; i < 3; i++)
            H[i] = MATRIX12::zeros();

        //crossed = cross(e2, e0);
        //crossNorm = norm(crossed);
        //crossGradient = cross_gradient(x);
        VECTOR3 crossed = e[2].cross(e[0]);
        MATRIX3x12 crossGrad = crossGradientVF(e);
        const VECTOR3& z = crossed;
        
        //denom15 = (z' * z) ^ (1.5);
        REAL denom15 = zs::pow(crossed.dot(crossed), (REAL)1.5);
        REAL denom25 = zs::pow(crossed.dot(crossed), (REAL)2.5);

        for (int j = 0; j < 12; j++)
            for (int i = 0; i < 12; i++)
            {
            auto zGradi = VECTOR3(crossGrad[0][i],crossGrad[1][i],crossGrad[2][i]);
            auto zGradj = VECTOR3(crossGrad[0][j],crossGrad[1][j],crossGrad[2][j]);
            VECTOR3 zHessianij = crossHessianVF(i,j);

            // z = cross(e2, e0);
            // zGrad = crossGradientVF(:,i);
            // alpha= (z' * zGrad) / (z' * z) ^ (1.5);
            REAL a = z.dot(zGradi) / denom15;

            // final = (zGradj' * zGradi) / denom15 + (z' * cross_hessian(i,j)) / denom15;
            // final = final - 3 * ((z' * zGradi) / denom25) * (zGradj' * z);
            REAL aGrad = (zGradj.dot(zGradi)) / denom15 + z.dot(crossHessianVF(i,j)) / denom15;
            aGrad -= (REAL)3.0 * (z.dot(zGradi) / denom25) * zGradj.dot(z);
            
            //entry = -((zGradj' * z) / denom15) * zGradi + 1 / norm(z) * zHessianij - alpha * zGradj - alphaGradj * z;
            VECTOR3 entry = -((zGradj.dot(z)) / denom15) * zGradi + (REAL)1.0 / z.norm() * zHessianij - a * zGradj - aGrad * z;

            H[0](i,j) = entry[0];
            H[1](i,j) = entry[1];
            H[2](i,j) = entry[2];
            }
        // return H;
    }


    ///////////////////////////////////////////////////////////////////////
    // hessian of spring length, n' * (v[2] - v[0])
    ///////////////////////////////////////////////////////////////////////
    constexpr MATRIX12 springLengthHessian(const VECTOR3 e[3],
                                                const VECTOR3& n,
                                                const VECTOR3& diff,
                                                const VECTOR2& a,
                                                const VECTOR2& b)
    {
        MATRIX3x12 tPartial = vDiffPartial(a,b);
        const REAL sign = (diff.dot(n) > (REAL)0.0) ? (REAL)-1.0 : (REAL)1.0;

        //% mode-3 contraction
        //[nx ny nz] = normal_hessian(x);
        //final = nx * delta(1) + ny * delta(2) + nz * delta(3);
        MATRIX12 normalH[3] = {};
        normalHessianEE(e,normalH);

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
    // hessian of spring length, n' * (v[0] - v[2])
    ///////////////////////////////////////////////////////////////////////
    constexpr MATRIX12 springLengthHessian(const VECTOR3 v[3],
                                                        const VECTOR3 e[3],
                                                        const VECTOR3& n){
        const VECTOR3 tvf = v[0] - v[2];

        MATRIX3x12 tvfPartial{0};
        // tvfPartial.setZero();
        tvfPartial(0,0) = tvfPartial(1,1) = tvfPartial(2,2) = 1.0;
        tvfPartial(0,6) = tvfPartial(1,7) = tvfPartial(2,8) = -1.0;

        //% mode-3 contraction
        //[nx ny nz] = normal_hessian(x);
        //final = nx * tvf(1) + ny * tvf(2) + nz * tvf(3);
        MATRIX12 normalH[3]={};
        normalHessianVF(e,normalH);
        const MATRIX12 contracted = tvf[0] * normalH[0] + tvf[1] * normalH[1] + 
                                    tvf[2] * normalH[2];
        
        const MATRIX3x12 nGrad = normalGradientVF(e);

        //product = nGrad' * vGrad;
        const MATRIX12 product = nGrad.transpose() * tvfPartial;

        return contracted + product + product.transpose();
    }

    ///////////////////////////////////////////////////////////////////////
    // get the linear interpolation coordinates from v0 to the line segment
    // between v1 and v2
    ///////////////////////////////////////////////////////////////////////
    constexpr VECTOR2 getLerp(const VECTOR3 v0, const VECTOR3& v1, const VECTOR3& v2)
    {
        const VECTOR3 e0 = v0 - v1;
        const VECTOR3 e1 = v2 - v1;
        const VECTOR3 e1hat = e1 / e1.norm();
        const REAL projection = e0.dot(e1hat);

        if (projection < 0.0)
            return VECTOR2(1.0, 0.0);

        if (projection >= e1.norm())
            return VECTOR2(0.0, 1.0);

        const REAL ratio = projection / e1.norm();
        return VECTOR2(1.0 - ratio, ratio);
    }


    ///////////////////////////////////////////////////////////////////////
    // find the distance from a line segment (v1, v2) to a point (v0)
    ///////////////////////////////////////////////////////////////////////
    constexpr REAL pointLineDistance(const VECTOR3 v0, const VECTOR3& v1, const VECTOR3& v2)
    {
        const VECTOR3 e0 = v0 - v1;
        const VECTOR3 e1 = v2 - v1;
        const VECTOR3 e1hat = e1 / e1.norm();
        const REAL projection = e0.dot(e1hat);

        // if it projects onto the line segment, use that length
        if (projection > 0.0 && projection < e1.norm())
        {
            const VECTOR3 normal = e0 - projection * e1hat;
            return normal.norm();
        }

        // if it doesn't, find the point-point distances
        const REAL diff01 = (v0 - v1).norm();
        const REAL diff02 = (v0 - v2).norm();

        return (diff01 < diff02) ? diff01 : diff02;
    }


    ///////////////////////////////////////////////////////////////////////
    // get the barycentric coordinate of the projection of v[0] onto the triangle
    // formed by v[1], v[2], v[3]
    ///////////////////////////////////////////////////////////////////////
    constexpr VECTOR3 getBarycentricCoordinates(const VECTOR3 vertices[4])
    {
        const VECTOR3 v0 = vertices[1];
        const VECTOR3 v1 = vertices[2];
        const VECTOR3 v2 = vertices[3];
            
        const VECTOR3 e1 = v1 - v0;
        const VECTOR3 e2 = v2 - v0;
        const VECTOR3 n = e1.cross(e2);
        const VECTOR3 nHat = n / n.norm();
        const VECTOR3 v = vertices[0] - (nHat.dot(vertices[0] - v0)) * nHat;

        // get the barycentric coordinates
        const VECTOR3 na = (v2 - v1).cross(v - v1);
        const VECTOR3 nb = (v0 - v2).cross(v - v2);
        const VECTOR3 nc = (v1 - v0).cross(v - v0);
        const VECTOR3 barycentric(n.dot(na) / n.l2NormSqr(),
                                    n.dot(nb) / n.l2NormSqr(),
                                    n.dot(nc) / n.l2NormSqr());

        return barycentric;
    }


    ///////////////////////////////////////////////////////////////////////
    // get the barycentric coordinate of the projection of v[0] onto the triangle
    // formed by v[1], v[2], v[3]
    //
    // but, if the projection is actually outside, project to all of the
    // edges and find the closest point that's still inside the triangle
    ///////////////////////////////////////////////////////////////////////
    constexpr VECTOR3 getInsideBarycentricCoordinates(const VECTOR3 vertices[4])
    {
        VECTOR3 barycentric = getBarycentricCoordinates(vertices);

        // if it's already inside, we're all done
        if (barycentric[0] >= 0.0 &&
            barycentric[1] >= 0.0 &&
            barycentric[2] >= 0.0)
            return barycentric;

        // find distance to all the line segments
        //
        // there's lots of redundant computation between here and getLerp,
        // but let's get it working and see if it fixes the actual
        // artifact before optimizing
        REAL distance12 = pointLineDistance(vertices[0], vertices[1], vertices[2]);
        REAL distance23 = pointLineDistance(vertices[0], vertices[2], vertices[3]);
        REAL distance31 = pointLineDistance(vertices[0], vertices[3], vertices[1]);

        // less than or equal is important here, otherwise fallthrough breaks
        if (distance12 <= distance23 && distance12 <= distance31)
        {
            VECTOR2 lerp = getLerp(vertices[0], vertices[1], vertices[2]);
            barycentric[0] = lerp[0];
            barycentric[1] = lerp[1];
            barycentric[2] = 0.0;
            return barycentric;
        }
        
        // less than or equal is important here, otherwise fallthrough breaks
        if (distance23 <= distance12 && distance23 <= distance31)
        {
            VECTOR2 lerp = getLerp(vertices[0], vertices[2], vertices[3]);
            barycentric[0] = 0.0;
            barycentric[1] = lerp[0];
            barycentric[2] = lerp[1];
            return barycentric;
        }

        // else it must be the 31 case
        VECTOR2 lerp = getLerp(vertices[0], vertices[3], vertices[1]);
        barycentric[0] = lerp[1];
        barycentric[1] = 0.0;
        barycentric[2] = lerp[0];
        return barycentric;
    }

    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    constexpr MATRIX3x12 tDiffPartial(const VECTOR3& bary)
    {
        MATRIX3x12 tPartial{0};
        tPartial(0,0) = tPartial(1,1)  = tPartial(2,2) = 1.0;
        tPartial(0,3) = tPartial(1,4)  = tPartial(2,5) = -bary[0];
        tPartial(0,6) = tPartial(1,7)  = tPartial(2,8) = -bary[1];
        tPartial(0,9) = tPartial(1,10) = tPartial(2,11) = -bary[2];

        return tPartial;
    }

    ///////////////////////////////////////////////////////////////////////
    // are the two edges nearly parallel?
    ///////////////////////////////////////////////////////////////////////
    constexpr bool nearlyParallel(const VECTOR3 e[3]){
        const VECTOR3 e0 = e[0].normalized();
        const VECTOR3 e1 = e[1].normalized();
        const REAL dotted = zs::abs(e0.dot(e1));

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

        return (dotted > (REAL)1.0 - eps);
    }

   


    ///////////////////////////////////////////////////////////////////////
    // does this face and edge intersect?
    ///////////////////////////////////////////////////////////////////////
    constexpr bool faceEdgeIntersection(const VECTOR3 triangleVertices[3], 
                            const VECTOR3 edgeVertices[2])
    {
        // assert(triangleVertices.size() == 3);
        // assert(edgeVertices.size() == 2);

        VECTOR3 a = triangleVertices[0];
        VECTOR3 b = triangleVertices[1];
        VECTOR3 c = triangleVertices[2];

        VECTOR3 origin = edgeVertices[0];
        VECTOR3 edgeDiff = (edgeVertices[1] - edgeVertices[0]);
        VECTOR3 direction = edgeDiff.normalized();

        VECTOR3 geometricNormal = ((b - a).cross(c - a)).normalized();

        VECTOR3 diff = a - origin;
        REAL denom = direction.dot(geometricNormal);
        if (zs::abs(denom) <= 0.0) return false;

        REAL t = diff.dot(geometricNormal) / denom;
        if (t < 0) return false;

        VECTOR3 h = origin + direction * t;

        VECTOR3 test = (b - a).cross(h - a);
        if (geometricNormal.dot(test) < 0) return false; 
        test = (c - b).cross(h - b);
        if (geometricNormal.dot(test) < 0) return false; 
        test = (a - c).cross(h - c);
        if (geometricNormal.dot(test) < 0) return false; 

        if (t < edgeDiff.norm())
            return true;

        return false;
    }


///////////////////////////////////////////////////////////////////////
// compute distance between a point and triangle
///////////////////////////////////////////////////////////////////////
    constexpr REAL pointTriangleDistance(const VECTOR3& v0, const VECTOR3& v1, 
                                        const VECTOR3& v2, const VECTOR3& v,VECTOR3& barycentric)
    {
        // get the barycentric coordinates
        const VECTOR3 e1 = v1 - v0;
        const VECTOR3 e2 = v2 - v0;
        const VECTOR3 n = e1.cross(e2);
        const VECTOR3 na = (v2 - v1).cross(v - v1);
        const VECTOR3 nb = (v0 - v2).cross(v - v2);
        const VECTOR3 nc = (v1 - v0).cross(v - v0);
        barycentric = VECTOR3(n.dot(na) / n.l2NormSqr(),
                                    n.dot(nb) / n.l2NormSqr(),
                                    n.dot(nc) / n.l2NormSqr());
                                    
        const REAL barySum = zs::abs(barycentric[0]) + zs::abs(barycentric[1]) + zs::abs(barycentric[2]);

        // if the point projects to inside the triangle, it should sum to 1
        if (zs::abs(barySum - 1.0) < 1e-6)
        {
            const VECTOR3 nHat = n / n.norm();
            const REAL normalDistance = (nHat.dot(v - v0));
            return zs::abs(normalDistance);
        }

        // project onto each edge, find the distance to each edge
        const VECTOR3 e3 = v2 - v1;
        const VECTOR3 ev = v - v0;
        const VECTOR3 ev3 = v - v1;
        const VECTOR3 e1Hat = e1 / e1.norm();
        const VECTOR3 e2Hat = e2 / e2.norm();
        const VECTOR3 e3Hat = e3 / e3.norm();
        VECTOR3 edgeDistances(1e8, 1e8, 1e8);

        // see if it projects onto the interval of the edge
        // if it doesn't, then the vertex distance will be smaller,
        // so we can skip computing anything
        const REAL e1dot = e1Hat.dot(ev);
        if (e1dot > 0.0 && e1dot < e1.norm())
        {
            const VECTOR3 projected = v0 + e1Hat * e1dot;
            edgeDistances[0] = (v - projected).norm();
        }
        const REAL e2dot = e2Hat.dot(ev);
        if (e2dot > 0.0 && e2dot < e2.norm())
        {
            const VECTOR3 projected = v0 + e2Hat * e2dot;
            edgeDistances[1] = (v - projected).norm();
        }
        const REAL e3dot = e3Hat.dot(ev3);
        if (e3dot > 0.0 && e3dot < e3.norm())
        {
            const VECTOR3 projected = v1 + e3Hat * e3dot;
            edgeDistances[2] = (v - projected).norm();
        }

        // get the distance to each vertex
        const VECTOR3 vertexDistances((v - v0).norm(), 
                                        (v - v1).norm(), 
                                        (v - v2).norm());

        // get the smallest of both the edge and vertex distances
        REAL vertexMin = 1e8;
        REAL edgeMin = 1e8;
        for(int i = 0;i < 3;++i){
            vertexMin = vertexMin > vertexDistances[i] ? vertexDistances[i] : vertexMin;
            edgeMin = edgeMin > edgeDistances[i] ? edgeDistances[i] : edgeMin;
        }
        // return the smallest of those
        return (vertexMin < edgeMin) ? vertexMin : edgeMin;
    }

constexpr REAL pointTriangleDistance(const VECTOR3& v0, const VECTOR3& v1, 
                                        const VECTOR3& v2, const VECTOR3& v)
    {
        // // get the barycentric coordinates
        // const VECTOR3 e1 = v1 - v0;
        // const VECTOR3 e2 = v2 - v0;
        // const VECTOR3 n = e1.cross(e2);
        // const VECTOR3 na = (v2 - v1).cross(v - v1);
        // const VECTOR3 nb = (v0 - v2).cross(v - v2);
        // const VECTOR3 nc = (v1 - v0).cross(v - v0);
        // const VECTOR3 barycentric(n.dot(na) / n.l2NormSqr(),
        //                             n.dot(nb) / n.l2NormSqr(),
        //                             n.dot(nc) / n.l2NormSqr());
                                    
        // const REAL barySum = zs::abs(barycentric[0]) + zs::abs(barycentric[1]) + zs::abs(barycentric[2]);

        // // if the point projects to inside the triangle, it should sum to 1
        // if (zs::abs(barySum - 1.0) < 1e-6)
        // {
        //     const VECTOR3 nHat = n / n.norm();
        //     const REAL normalDistance = (nHat.dot(v - v0));
        //     return zs::abs(normalDistance);
        // }

        // // project onto each edge, find the distance to each edge
        // const VECTOR3 e3 = v2 - v1;
        // const VECTOR3 ev = v - v0;
        // const VECTOR3 ev3 = v - v1;
        // const VECTOR3 e1Hat = e1 / e1.norm();
        // const VECTOR3 e2Hat = e2 / e2.norm();
        // const VECTOR3 e3Hat = e3 / e3.norm();
        // VECTOR3 edgeDistances(1e8, 1e8, 1e8);

        // // see if it projects onto the interval of the edge
        // // if it doesn't, then the vertex distance will be smaller,
        // // so we can skip computing anything
        // const REAL e1dot = e1Hat.dot(ev);
        // if (e1dot > 0.0 && e1dot < e1.norm())
        // {
        //     const VECTOR3 projected = v0 + e1Hat * e1dot;
        //     edgeDistances[0] = (v - projected).norm();
        // }
        // const REAL e2dot = e2Hat.dot(ev);
        // if (e2dot > 0.0 && e2dot < e2.norm())
        // {
        //     const VECTOR3 projected = v0 + e2Hat * e2dot;
        //     edgeDistances[1] = (v - projected).norm();
        // }
        // const REAL e3dot = e3Hat.dot(ev3);
        // if (e3dot > 0.0 && e3dot < e3.norm())
        // {
        //     const VECTOR3 projected = v1 + e3Hat * e3dot;
        //     edgeDistances[2] = (v - projected).norm();
        // }

        // // get the distance to each vertex
        // const VECTOR3 vertexDistances((v - v0).norm(), 
        //                                 (v - v1).norm(), 
        //                                 (v - v2).norm());

        // // get the smallest of both the edge and vertex distances
        // REAL vertexMin = 1e8;
        // REAL edgeMin = 1e8;
        // for(int i = 0;i < 3;++i){
        //     vertexMin = vertexMin > vertexDistances[i] ? vertexDistances[i] : vertexMin;
        //     edgeMin = edgeMin > edgeDistances[i] ? edgeDistances[i] : edgeMin;
        // }
        // // return the smallest of those
        // return (vertexMin < edgeMin) ? vertexMin : edgeMin;
        VECTOR3 barycentric{};
        return pointTriangleDistance(v0,v1,v2,v,barycentric);
    }



constexpr REAL pointTriangleDistance(const VECTOR3& v0, const VECTOR3& v1, 
                                        const VECTOR3& v2, const VECTOR3& v,REAL& barySum)
    {
        // get the barycentric coordinates
        const VECTOR3 e1 = v1 - v0;
        const VECTOR3 e2 = v2 - v0;
        const VECTOR3 n = e1.cross(e2);
        const VECTOR3 na = (v2 - v1).cross(v - v1);
        const VECTOR3 nb = (v0 - v2).cross(v - v2);
        const VECTOR3 nc = (v1 - v0).cross(v - v0);
        const VECTOR3 barycentric(n.dot(na) / n.l2NormSqr(),
                                    n.dot(nb) / n.l2NormSqr(),
                                    n.dot(nc) / n.l2NormSqr());
                                    
        barySum = zs::abs(barycentric[0]) + zs::abs(barycentric[1]) + zs::abs(barycentric[2]);

        // if the point projects to inside the triangle, it should sum to 1
        if (zs::abs(barySum - 1.0) < 1e-6)
        {
            const VECTOR3 nHat = n / n.norm();
            const REAL normalDistance = (nHat.dot(v - v0));
            return zs::abs(normalDistance);
        }

        // project onto each edge, find the distance to each edge
        const VECTOR3 e3 = v2 - v1;
        const VECTOR3 ev = v - v0;
        const VECTOR3 ev3 = v - v1;
        const VECTOR3 e1Hat = e1 / e1.norm();
        const VECTOR3 e2Hat = e2 / e2.norm();
        const VECTOR3 e3Hat = e3 / e3.norm();
        VECTOR3 edgeDistances(1e8, 1e8, 1e8);

        // see if it projects onto the interval of the edge
        // if it doesn't, then the vertex distance will be smaller,
        // so we can skip computing anything
        const REAL e1dot = e1Hat.dot(ev);
        if (e1dot > 0.0 && e1dot < e1.norm())
        {
            const VECTOR3 projected = v0 + e1Hat * e1dot;
            edgeDistances[0] = (v - projected).norm();
        }
        const REAL e2dot = e2Hat.dot(ev);
        if (e2dot > 0.0 && e2dot < e2.norm())
        {
            const VECTOR3 projected = v0 + e2Hat * e2dot;
            edgeDistances[1] = (v - projected).norm();
        }
        const REAL e3dot = e3Hat.dot(ev3);
        if (e3dot > 0.0 && e3dot < e3.norm())
        {
            const VECTOR3 projected = v1 + e3Hat * e3dot;
            edgeDistances[2] = (v - projected).norm();
        }

        // get the distance to each vertex
        const VECTOR3 vertexDistances((v - v0).norm(), 
                                        (v - v1).norm(), 
                                        (v - v2).norm());

        // get the smallest of both the edge and vertex distances
        REAL vertexMin = 1e8;
        REAL edgeMin = 1e8;
        for(int i = 0;i < 3;++i){
            vertexMin = vertexMin > vertexDistances[i] ? vertexDistances[i] : vertexMin;
            edgeMin = edgeMin > edgeDistances[i] ? edgeDistances[i] : edgeMin;
        }
        // return the smallest of those
        return (vertexMin < edgeMin) ? vertexMin : edgeMin;
    }


    ///////////////////////////////////////////////////////////////////////
    // see if the projection of v onto the plane of v0,v1,v2 is inside 
    // the triangle formed by v0,v1,v2
    ///////////////////////////////////////////////////////////////////////
    constexpr bool pointProjectsInsideTriangle(const VECTOR3& v0, const VECTOR3& v1, 
                                            const VECTOR3& v2, const VECTOR3& v){
        // get the barycentric coordinates
        const VECTOR3 e1 = v1 - v0;
        const VECTOR3 e2 = v2 - v0;
        const VECTOR3 n = e1.cross(e2);
        const VECTOR3 na = (v2 - v1).cross(v - v1);
        const VECTOR3 nb = (v0 - v2).cross(v - v2);
        const VECTOR3 nc = (v1 - v0).cross(v - v0);
        const VECTOR3 barycentric(n.dot(na) / n.l2NormSqr(),
                                    n.dot(nb) / n.l2NormSqr(),
                                    n.dot(nc) / n.l2NormSqr());
                                    
        const REAL barySum = zs::abs(barycentric[0]) + zs::abs(barycentric[1]) + zs::abs(barycentric[2]);

        // if the point projects to inside the triangle, it should sum to 1
        if (zs::abs(barySum - 1.0) < 1e-6)
            return true;

        return false;
    }

    #define FMAX(a,b) ((a) > (b) ? (a) : (b))
    #define FMIN(a,b) ((a) > (b) ? (b) : (a))
    #define FABS(a) ((a) < 0.0f ? -(a) : (a))
    #define OUT_OF_RANGE(a) ((a) < 0.0f || (a) > 1.f)


    /**************************************************************************
    |
    |     Method: FindNearestPointOnLineSegment
    |
    |    Purpose: Given a line (segment) and a point in 3-dimensional space,
    |             find the point on the line (segment) that is closest to the
    |             point.
    |
    | Parameters: Input:
    |             ------
    |             A1x, A1y, A1z   - Coordinates of first defining point of the line/segment
    |             Lx, Ly, Lz      - Vector from (A1x, A1y, A1z) to the second defining point
    |                               of the line/segment.
    |             Bx, By, Bz      - Coordinates of the point
    |             infinite_lines  - set to true if lines are to be treated as infinite
    |             epsilon_squared - tolerance value to be used to check for degenerate
    |                               and parallel lines, and to check for true intersection.
    |
    |             Output:
    |             -------
    |             NearestPointX,  - Point on line/segment that is closest to (Bx, By, Bz)
    |             NearestPointY,
    |             NearestPointZ
    |             parameter       - Parametric coordinate of the nearest point along the
    |                               line/segment. parameter = 0 at (A1x, A1y, A1z) and
    |                               parameter = 1 at the second defining point of the line/
    |                               segmetn
    **************************************************************************/
    constexpr void FindNearestPointOnLineSegment(const REAL A1x, const REAL A1y, const REAL A1z,
                                    const REAL Lx, const REAL Ly, const REAL Lz,
                                    const REAL Bx, const REAL By, const REAL Bz,
                                    bool infinite_line, REAL epsilon_squared, REAL &NearestPointX,
                                    REAL &NearestPointY, REAL &NearestPointZ,
                                    REAL &parameter)
    {
        // Line/Segment is degenerate --- special case #1
        REAL D = Lx * Lx + Ly * Ly + Lz * Lz;
        if (D < epsilon_squared)
        {
            NearestPointX = A1x;
            NearestPointY = A1y;
            NearestPointZ = A1z;
            return;
        }

        REAL ABx = Bx - A1x;
        REAL ABy = By - A1y;
        REAL ABz = Bz - A1z;

        // parameter is computed from Equation (20).
        parameter = (Lx * ABx + Ly * ABy + Lz * ABz) / D;

        if (false == infinite_line) parameter = (REAL)FMAX(0.0, FMIN(1.0, parameter));

        NearestPointX = A1x + parameter * Lx;
        NearestPointY = A1y + parameter * Ly;
        NearestPointZ = A1z + parameter * Lz;
        return;
    }


    /**************************************************************************
    |
    |     Method: AdjustNearestPoints
    |
    |    Purpose: Given nearest point information for two infinite lines, adjust
    |             to model finite line segments.
    |
    | Parameters: Input:
    |             ------
    |             A1x, A1y, A1z   - Coordinates of first defining point of line/segment A
    |             Lax, Lay, Laz   - Vector from (A1x, A1y, A1z) to the (A2x, A2y, A2z).
    |             B1x, B1y, B1z   - Coordinates of first defining point of line/segment B
    |             Lbx, Lby, Lbz   - Vector from (B1x, B1y, B1z) to the (B2x, B2y, B2z).
    |             epsilon_squared - tolerance value to be used to check for degenerate
    |                               and parallel lines, and to check for true intersection.
    |             s               - parameter representing nearest point on infinite line A
    |             t               - parameter representing nearest point on infinite line B
    |
    |             Output:
    |             -------
    |             PointOnSegAx,   - Coordinates of the point on segment A that are nearest
    |             PointOnSegAy,     to segment B. This corresponds to point C in the text.
    |             PointOnSegAz
    |             PointOnSegBx,   - Coordinates of the point on segment B that are nearest
    |             PointOnSegBy,     to segment A. This corresponds to point D in the text.
    |             PointOnSegBz
    **************************************************************************/
    constexpr void AdjustNearestPoints(REAL A1x, REAL A1y, REAL A1z,
                            REAL Lax, REAL Lay, REAL Laz,
                            REAL B1x, REAL B1y, REAL B1z,
                            REAL Lbx, REAL Lby, REAL Lbz,
                            REAL epsilon_squared, REAL s, REAL t,
                            REAL &PointOnSegAx, REAL &PointOnSegAy, REAL &PointOnSegAz,
                            REAL &PointOnSegBx, REAL &PointOnSegBy, REAL &PointOnSegBz)
    {
    // handle the case where both parameter s and t are out of range
        if (OUT_OF_RANGE(s) && OUT_OF_RANGE(t))
        {
            s = FMAX((REAL)0.0, FMIN((REAL)1.0, s));
            PointOnSegAx = (A1x + s * Lax);
            PointOnSegAy = (A1y + s * Lay);
            PointOnSegAz = (A1z + s * Laz);
            FindNearestPointOnLineSegment(B1x, B1y, B1z, Lbx, Lby, Lbz, PointOnSegAx,
                                        PointOnSegAy, PointOnSegAz, true, epsilon_squared,
                                        PointOnSegBx, PointOnSegBy, PointOnSegBz, t);
            if (OUT_OF_RANGE(t))
            {
                t = FMAX((REAL)0.0, FMIN((REAL)1.0, t));
                PointOnSegBx = (B1x + t * Lbx);
                PointOnSegBy = (B1y + t * Lby);
                PointOnSegBz = (B1z + t * Lbz);
                FindNearestPointOnLineSegment(A1x, A1y, A1z, Lax, Lay, Laz, PointOnSegBx,
                                                PointOnSegBy, PointOnSegBz, false, epsilon_squared,
                                                PointOnSegAx, PointOnSegAy, PointOnSegAz, s);
                FindNearestPointOnLineSegment(B1x, B1y, B1z, Lbx, Lby, Lbz, PointOnSegAx,
                                                PointOnSegAy, PointOnSegAz, false, epsilon_squared,
                                                PointOnSegBx, PointOnSegBy, PointOnSegBz, t);
            }
        }
        // otherwise, handle the case where the parameter for only one segment is
        // out of range
        else if (OUT_OF_RANGE(s))
        {
            s = FMAX((REAL)0.0, FMIN((REAL)1.0, s));
            PointOnSegAx = (A1x + s * Lax);
            PointOnSegAy = (A1y + s * Lay);
            PointOnSegAz = (A1z + s * Laz);
            FindNearestPointOnLineSegment(B1x, B1y, B1z, Lbx, Lby, Lbz, PointOnSegAx,
                                        PointOnSegAy, PointOnSegAz, false, epsilon_squared,
                                        PointOnSegBx, PointOnSegBy, PointOnSegBz, t);
        }
        else if (OUT_OF_RANGE(t))
        {
            t = FMAX((REAL)0.0, FMIN((REAL)1.0, t));
            PointOnSegBx = (B1x + t * Lbx);
            PointOnSegBy = (B1y + t * Lby);
            PointOnSegBz = (B1z + t * Lbz);
            FindNearestPointOnLineSegment(A1x, A1y, A1z, Lax, Lay, Laz, PointOnSegBx,
                                        PointOnSegBy, PointOnSegBz, false, epsilon_squared,
                                        PointOnSegAx, PointOnSegAy, PointOnSegAz, s);
        }
    }    


    /**************************************************************************
    |
    |     Method: FindNearestPointOfParallelLineSegments
    |
    |    Purpose: Given two lines (segments) that are known to be parallel, find
    |             a representative point on each that is nearest to the other. If
    |             the lines are considered to be finite then it is possible that there
    |             is one true point on each line that is nearest to the other. This
    |             code properly handles this case.
    |
    |             This is the most difficult line intersection case to handle, since
    |             there is potentially a family, or locus of points on each line/segment
    |             that are nearest to the other.
    | Parameters: Input:
    |             ------
    |             A1x, A1y, A1z   - Coordinates of first defining point of line/segment A
    |             A2x, A2y, A2z   - Coordinates of second defining point of line/segment A
    |             Lax, Lay, Laz   - Vector from (A1x, A1y, A1z) to the (A2x, A2y, A2z).
    |             B1x, B1y, B1z   - Coordinates of first defining point of line/segment B
    |             B2x, B2y, B2z   - Coordinates of second defining point of line/segment B
    |             Lbx, Lby, Lbz   - Vector from (B1x, B1y, B1z) to the (B2x, B2y, B2z).
    |             infinite_lines  - set to true if lines are to be treated as infinite
    |             epsilon_squared - tolerance value to be used to check for degenerate
    |                               and parallel lines, and to check for true intersection.
    |
    |             Output:
    |             -------
    |             PointOnSegAx,   - Coordinates of the point on segment A that are nearest
    |             PointOnSegAy,     to segment B. This corresponds to point C in the text.
    |             PointOnSegAz
    |             PointOnSegBx,   - Coordinates of the point on segment B that are nearest
    |             PointOnSegBy,     to segment A. This corresponds to point D in the text.
    |             PointOnSegBz

    **************************************************************************/
    constexpr void FindNearestPointOfParallelLineSegments(REAL A1x, REAL A1y, REAL A1z,
                                                REAL A2x, REAL A2y, REAL A2z,
                                                REAL Lax, REAL Lay, REAL Laz,
                                                REAL B1x, REAL B1y, REAL B1z,
                                                REAL B2x, REAL B2y, REAL B2z,
                                                REAL Lbx, REAL Lby, REAL Lbz,
                                                bool infinite_lines, REAL epsilon_squared,
                                                REAL &PointOnSegAx, REAL &PointOnSegAy, REAL &PointOnSegAz,
                                                REAL &PointOnSegBx, REAL &PointOnSegBy, REAL &PointOnSegBz)
    {
        REAL s[2] = {0, 0};
        REAL temp{};
        FindNearestPointOnLineSegment(A1x, A1y, A1z, Lax, Lay, Laz, B1x, B1y, B1z,
                                        true, epsilon_squared, PointOnSegAx, PointOnSegAy, PointOnSegAz, s[0]);
        if (true == infinite_lines)
        {
            PointOnSegBx = B1x;
            PointOnSegBy = B1y;
            PointOnSegBz = B1z;
        }
        else
        {
            REAL tp[3] = {};
            FindNearestPointOnLineSegment(A1x, A1y, A1z, Lax, Lay, Laz, B2x, B2y, B2z,
                                        true, epsilon_squared, tp[0], tp[1], tp[2], s[1]);
            if (s[0] < 0.0 && s[1] < 0.0)
            {
                PointOnSegAx = A1x;
                PointOnSegAy = A1y;
                PointOnSegAz = A1z;
                if (s[0] < s[1])
                {
                    PointOnSegBx = B2x;
                    PointOnSegBy = B2y;
                    PointOnSegBz = B2z;
                }
                else
                {
                    PointOnSegBx = B1x;
                    PointOnSegBy = B1y;
                    PointOnSegBz = B1z;
                }
            }
            else if (s[0] > (REAL)1.0 && s[1] > (REAL)1.0)
            {
                PointOnSegAx = A2x;
                PointOnSegAy = A2y;
                PointOnSegAz = A2z;
                if (s[0] < s[1])
                {
                    PointOnSegBx = B1x;
                    PointOnSegBy = B1y;
                    PointOnSegBz = B1z;
                }
                else
                {
                    PointOnSegBx = B2x;
                    PointOnSegBy = B2y;
                    PointOnSegBz = B2z;
                }
            }
            else
            {
                temp = (REAL)0.5*(FMAX((REAL)0.0, FMIN((REAL)1.0, s[0])) + FMAX((REAL)0.0, FMIN((REAL)1.0, s[1])));
                PointOnSegAx = (A1x + temp * Lax);
                PointOnSegAy = (A1y + temp * Lay);
                PointOnSegAz = (A1z + temp * Laz);
                FindNearestPointOnLineSegment(B1x, B1y, B1z, Lbx, Lby, Lbz,
                                                PointOnSegAx, PointOnSegAy, PointOnSegAz, true,
                                                epsilon_squared, PointOnSegBx, PointOnSegBy, PointOnSegBz, temp);
            }
        }
    }



    /**************************************************************************
    |
    |     Method: IntersectLineSegments
    |
    |    Purpose: Find the nearest point between two finite length line segments
    |             or two infinite lines in 3-dimensional space. The function calculates
    |             the point on each line/line segment that is closest to the other
    |             line/line segment, the midpoint between the nearest points, and
    |             the vector between these two points. If the two nearest points
    |             are close within a tolerance, a flag is set indicating the lines
    |             have a "true" intersection.
    |
    | Parameters: Input:
    |             ------
    |             A1x, A1y, A1z   - Coordinates of first defining point of line/segment A
    |             A2x, A2y, A2z   - Coordinates of second defining point of line/segment A
    |             B1x, B1y, B1z   - Coordinates of first defining point of line/segment B
    |             B2x, B2y, B2z   - Coordinates of second defining point of line/segment B
    |             infinite_lines  - set to true if lines are to be treated as infinite
    |             epsilon         - tolerance value to be used to check for degenerate
    |                               and parallel lines, and to check for true intersection.
    |
    |             Output:
    |             -------
    |             PointOnSegAx,   - Coordinates of the point on segment A that are nearest
    |             PointOnSegAy,     to segment B. This corresponds to point C in the text.
    |             PointOnSegAz
    |             PointOnSegBx,   - Coordinates of the point on segment B that are nearest
    |             PointOnSegBy,     to segment A. This corresponds to point D in the text.
    |             PointOnSegBz
    |             NearestPointX,  - Midpoint between the two nearest points. This can be
    |             NearestPointY,    treated as *the* intersection point if nearest points
    |             NearestPointZ     are sufficiently close. This corresponds to point P
    |                               in the text.
    |             NearestVectorX, - Vector between the nearest point on A to the nearest
    |                               point on segment B. This vector is normal to both
    |                               lines if the lines are infinite, but is not guaranteed
    |                               to be normal to both lines if both lines are finite
    |                               length.
    |           true_intersection - true if the nearest points are close within a small
    |                               tolerance.
    **************************************************************************/
    constexpr void IntersectLineSegments(const REAL A1x, const REAL A1y, const REAL A1z,
                            const REAL A2x, const REAL A2y, const REAL A2z,
                            const REAL B1x, const REAL B1y, const REAL B1z,
                            const REAL B2x, const REAL B2y, const REAL B2z,
                            bool infinite_lines, REAL epsilon, REAL &PointOnSegAx,
                            REAL &PointOnSegAy, REAL &PointOnSegAz, REAL &PointOnSegBx,
                            REAL &PointOnSegBy, REAL &PointOnSegBz, REAL &NearestPointX,
                            REAL &NearestPointY, REAL &NearestPointZ, REAL &NearestVectorX,
                            REAL &NearestVectorY, REAL &NearestVectorZ, bool &true_intersection)
    {
        REAL temp = (REAL)0.0;
        REAL epsilon_squared = epsilon * epsilon;

        // Compute parameters from Equations (1) and (2) in the text
        REAL Lax = A2x - A1x;
        REAL Lay = A2y - A1y;
        REAL Laz = A2z - A1z;
        REAL Lbx = B2x - B1x;
        REAL Lby = B2y - B1y;
        REAL Lbz = B2z - B1z;
        // From Equation (15)
        REAL L11 =  (Lax * Lax) + (Lay * Lay) + (Laz * Laz);
        REAL L22 =  (Lbx * Lbx) + (Lby * Lby) + (Lbz * Lbz);

        // Line/Segment A is degenerate ---- Special Case #1
        if (L11 < epsilon_squared)
        {
            PointOnSegAx = A1x;
            PointOnSegAy = A1y;
            PointOnSegAz = A1z;
            FindNearestPointOnLineSegment(B1x, B1y, B1z, Lbx, Lby, Lbz, A1x, A1y, A1z,
                                        infinite_lines, epsilon, PointOnSegBx, PointOnSegBy,
                                        PointOnSegBz, temp);
        }
        // Line/Segment B is degenerate ---- Special Case #1
        else if (L22 < epsilon_squared)
        {
            PointOnSegBx = B1x;
            PointOnSegBy = B1y;
            PointOnSegBz = B1z;
            FindNearestPointOnLineSegment(A1x, A1y, A1z, Lax, Lay, Laz, B1x, B1y, B1z,
                                        infinite_lines, epsilon, PointOnSegAx, PointOnSegAy,
                                        PointOnSegAz, temp);
        }
        // Neither line/segment is degenerate
        else
        {
            // Compute more parameters from Equation (3) in the text.
            REAL ABx = B1x - A1x;
            REAL ABy = B1y - A1y;
            REAL ABz = B1z - A1z;

            // and from Equation (15).
            REAL L12 = -(Lax * Lbx) - (Lay * Lby) - (Laz * Lbz);

            REAL DetL = L11 * L22 - L12 * L12;
            // Lines/Segments A and B are parallel ---- special case #2.
            if (FABS(DetL) < epsilon)
            {
                FindNearestPointOfParallelLineSegments(A1x, A1y, A1z, A2x, A2y, A2z,
                                                        Lax, Lay, Laz,
                                                        B1x, B1y, B1z, B2x, B2y, B2z,
                                                        Lbx, Lby, Lbz,
                                                        infinite_lines, epsilon,
                                                        PointOnSegAx, PointOnSegAy, PointOnSegAz,
                                                        PointOnSegBx, PointOnSegBy, PointOnSegBz);
            }
            // The general case
            else
            {
                // from Equation (15)
                REAL ra = Lax * ABx + Lay * ABy + Laz * ABz;
                REAL rb = -Lbx * ABx - Lby * ABy - Lbz * ABz;

                REAL t = (L11 * rb - ra * L12)/DetL; // Equation (12)

            #ifdef USE_CRAMERS_RULE
                REAL s = (L22 * ra - rb * L12)/DetL;
            #else
                REAL s = (ra-L12*t)/L11;             // Equation (13)
            #endif // USE_CRAMERS_RULE

            #ifdef CHECK_ANSWERS
                REAL check_ra = s*L11 + t*L12;
                REAL check_rb = s*L12 + t*L22;
                // assert(FABS(check_ra-ra) < epsilon);
                // assert(FABS(check_rb-rb) < epsilon);
            #endif // CHECK_ANSWERS

            // if we are dealing with infinite lines or if parameters s and t both
            // lie in the range [0,1] then just compute the points using Equations
            // (1) and (2) from the text.
                PointOnSegAx = (A1x + s * Lax);
                PointOnSegAy = (A1y + s * Lay);
                PointOnSegAz = (A1z + s * Laz);
                PointOnSegBx = (B1x + t * Lbx);
                PointOnSegBy = (B1y + t * Lby);
                PointOnSegBz = (B1z + t * Lbz);
            // otherwise, at least one of s and t is outside of [0,1] and we have to
            // handle this case.
                if (false == infinite_lines && (OUT_OF_RANGE(s) || OUT_OF_RANGE(t)))
                {
                    AdjustNearestPoints(A1x, A1y, A1z, Lax, Lay, Laz,
                                        B1x, B1y, B1z, Lbx, Lby, Lbz,
                                        epsilon, s, t,
                                        PointOnSegAx, PointOnSegAy, PointOnSegAz,
                                        PointOnSegBx, PointOnSegBy, PointOnSegBz);
                }
            }
        }

        NearestPointX = (REAL)0.5 * (PointOnSegAx + PointOnSegBx);
        NearestPointY = (REAL)0.5 * (PointOnSegAy + PointOnSegBy);
        NearestPointZ = (REAL)0.5 * (PointOnSegAz + PointOnSegBz);

        NearestVectorX = PointOnSegBx - PointOnSegAx;
        NearestVectorY = PointOnSegBy - PointOnSegAy;
        NearestVectorZ = PointOnSegBz - PointOnSegAz;

        // optional check to indicate if the lines truly intersect
        true_intersection = (FABS(NearestVectorX) +
                            FABS(NearestVectorY) +
                            FABS(NearestVectorZ)) < epsilon ? true : false;
    }


    constexpr void IntersectLineSegments(const VECTOR3& a0, const VECTOR3& a1,
                            const VECTOR3& b0, const VECTOR3& b1,
                            VECTOR3& aPoint, VECTOR3& bPoint)
    {
        VECTOR3 midpoint{};
        VECTOR3 normal{};
        bool intersect{};
        IntersectLineSegments(a0[0], a0[1], a0[2], a1[0], a1[1], a1[2],
                                b0[0], b0[1], b0[2], b1[0], b1[1], b1[2],
                                false, 1e-6,
                                aPoint[0], aPoint[1], aPoint[2],
                                bPoint[0], bPoint[1], bPoint[2],
                                midpoint[0], midpoint[1], midpoint[2],
                                normal[0], normal[1], normal[2], intersect);
    }

};
};