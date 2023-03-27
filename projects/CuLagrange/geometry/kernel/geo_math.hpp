#pragma once

#include <array>
#include <vector>

namespace zeno { namespace LSL_GEO {

    using REAL = float;
    using VECTOR12 = typename zs::vec<REAL,12>;
    using VECTOR4 = typename zs::vec<REAL,4>;
    using VECTOR3 = typename zs::vec<REAL,3>;
    using VECTOR2 = typename zs::vec<REAL,2>;
    using MATRIX3x12 = typename zs::vec<REAL,3,12>;
    using MATRIX12 = typename zs::vec<REAL,12,12>;

    template<int simplex_size,int ne = (simplex_size - 1) * simplex_size, zs::enable_if_t<(simplex_size >= 2 && simplex_size <= 4)> = 0>
    constexpr zs::vec<int,ne * 2> ordered_edges() {
        if constexpr (simplex_size == 4)    
            return zs::vec<int,ne * 2>{1,2,2,0,0,1,3,0,3,1,3,2};
        if constexpr (simplex_size == 3)
            return zs::vec<int,ne * 2>{1,2,2,0,0,1};
        if constexpr (simplex_size == 2)
            return zs::vec<int,ne * 2>{0,1};
    }

    template<typename T>
    constexpr T doublearea(T a,T b,T c) {
        T s = (a + b + c)/2;
        return 2*zs::sqrt(s*(s-a)*(s-b)*(s-c));
    }    

    // using T = float;
    template<typename VecT, zs::enable_if_all<VecT::dim == 1, (VecT::extent <= 3), (VecT::extent > 1)> = 0>
    constexpr auto area(const zs::VecInterface<VecT>& p0,const zs::VecInterface<VecT>& p1,const zs::VecInterface<VecT>& p2){
        auto a = (p0 - p1).norm();
        auto b = (p0 - p2).norm();
        auto c = (p1 - p2).norm();
        return doublearea(a,b,c) / (typename VecT::value_type)2.0;
    }
    // template<typename T,typename V = typename zs::vec<T,3>>
    // constexpr T area(const V p[3]){
    //     auto a = (p[0] - p[1]).norm();
    //     auto b = (p[0] - p[2]).norm();
    //     auto c = (p[1] - p[2]).norm();
    //     return doublearea(a,b,c) / 2.0;
    // }
    // template<typename T,zs::enable_if_t<std::is_integral_v<T>> = 0>
    // constexpr T area(T a,T b,T c) {
    //     return doublearea(a,b,c)/2;
    // }    


    template<typename T>
    constexpr T volume(const zs::vec<T, 6>& l) {
        T u = l(0);
        T v = l(1);
        T w = l(2);
        T U = l(3);
        T V = l(4);
        T W = l(5);
        T X = (w - U + v)*(U + v + w);
        T x = (U - v + w)*(v - w + U);
        T Y = (u - V + w)*(V + w + u);
        T y = (V - w + u)*(w - u + V);
        T Z = (v - W + u)*(W + u + v);
        T z = (W - u + v)*(u - v + W);
        T a = zs::sqrt(x*Y*Z);
        T b = zs::sqrt(y*Z*X);
        T c = zs::sqrt(z*X*Y);
        T d = zs::sqrt(x*y*z);
        T vol = zs::sqrt(
        (-a + b + c + d)*
        ( a - b + c + d)*
        ( a + b - c + d)*
        ( a + b + c - d))/
        (192.*u*v*w);

        return vol;
    }

    template<typename T,typename V = typename zs::vec<T,3>>
    constexpr T volume(const V p[4]) {
        zs::vec<T,6> l{};
        auto edges = ordered_edges<4>();
        for(size_t i= 0;i < 6;++i)
            l[i] = (p[edges[i*2 + 0]] - p[edges[i*2 + 1]]).norm();
        return volume<T>(l);
    }


    template<typename T,typename V = typename zs::vec<T,3>>
    constexpr T volume(const V& p0,const V& p1,const V& p2,const V& p3) {
        V p[4];
        p[0] = p0;p[1] = p1;p[2] = p2;p[3] = p3;
        return volume<T>(p);
    }




    // template<typename T>
    // constexpr T det(zs::vec<zs::vec<T,3>,4>& p) {
    //     return volume(p);
    // }

    template<typename T,int simplex_size,int space_dim,zs::enable_if_t<(space_dim == 3)> = 0>
    constexpr T det(zs::vec<zs::vec<T,space_dim>,simplex_size>& p) {
        if constexpr(simplex_size == 4)
            return volume(p);
        if constexpr(simplex_size == 3)
            return area(p);
        if constexpr(simplex_size == 2)
            return (p[0] - p[1]).norm();
    }

    template<typename T>
    constexpr void dihedral_angle_intrinsic(const zs::vec<T, 6>& l,const zs::vec<T, 4>& s,zs::vec<T, 6>& theta,zs::vec<T, 6>& cos_theta) {
        zs::vec<T, 6> H_sqr{};
        H_sqr[0] = (1./16.) * (4.*l(3)*l(3)*l(0)*l(0) - zs::sqr((l(1)*l(1) + l(4)*l(4)) - (l(2)*l(2) + l(5)*l(5))));
        H_sqr[1] = (1./16.) * (4.*l(4)*l(4)*l(1)*l(1) - zs::sqr((l(2)*l(2) + l(5)*l(5)) - (l(3)*l(3) + l(0)*l(0))));
        H_sqr[2] = (1./16.) * (4.*l(5)*l(5)*l(2)*l(2) - zs::sqr((l(3)*l(3) + l(0)*l(0)) - (l(4)*l(4) + l(1)*l(1))));
        H_sqr[3] = (1./16.) * (4.*l(0)*l(0)*l(3)*l(3) - zs::sqr((l(4)*l(4) + l(1)*l(1)) - (l(5)*l(5) + l(2)*l(2))));
        H_sqr[4] = (1./16.) * (4.*l(1)*l(1)*l(4)*l(4) - zs::sqr((l(5)*l(5) + l(2)*l(2)) - (l(0)*l(0) + l(3)*l(3))));
        H_sqr[5] = (1./16.) * (4.*l(2)*l(2)*l(5)*l(5) - zs::sqr((l(0)*l(0) + l(3)*l(3)) - (l(1)*l(1) + l(4)*l(4))));

        cos_theta(0) = (H_sqr(0) - s(1)*s(1) - s(2)*s(2)) / (-2.*s(1) * s(2));
        cos_theta(1) = (H_sqr(1) - s(2)*s(2) - s(0)*s(0)) / (-2.*s(2) * s(0));
        cos_theta(2) = (H_sqr(2) - s(0)*s(0) - s(1)*s(1)) / (-2.*s(0) * s(1));
        cos_theta(3) = (H_sqr(3) - s(3)*s(3) - s(0)*s(0)) / (-2.*s(3) * s(0));
        cos_theta(4) = (H_sqr(4) - s(3)*s(3) - s(1)*s(1)) / (-2.*s(3) * s(1));
        cos_theta(5) = (H_sqr(5) - s(3)*s(3) - s(2)*s(2)) / (-2.*s(3) * s(2));

        //TODO the theta here might be invalid, might be a hidden bug
        theta(0) = zs::acos(cos_theta(0));  
        theta(1) = zs::acos(cos_theta(1)); 
        theta(2) = zs::acos(cos_theta(2)); 
        theta(3) = zs::acos(cos_theta(3)); 
        theta(4) = zs::acos(cos_theta(4)); 
        theta(5) = zs::acos(cos_theta(5));       
    }

    template<typename T>
    constexpr zs::vec<T,3,3> deformation_gradient(
        const zs::vec<T,3>& x0,const zs::vec<T,3>& x1,const zs::vec<T,3>& x2,const zs::vec<T,3>& x3,const zs::vec<T,3,3>& IB) {
            auto x01 = x1 - x0;
            auto x02 = x2 - x0;
            auto x03 = x3 - x0;
            zs::vec<T,3,3> Dx{
                x01[0],x02[0],x03[0],
                x01[1],x02[1],x03[1],
                x01[2],x02[2],x03[2]};  
            return Dx * IB;
    }

    template<typename T>
    constexpr zs::vec<T,3,3> deformation_gradient(
            const zs::vec<T,3>& X0,const zs::vec<T,3>& X1,const zs::vec<T,3>& X2,const zs::vec<T,3>& X3,
            const zs::vec<T,3>& x0,const zs::vec<T,3>& x1,const zs::vec<T,3>& x2,const zs::vec<T,3>& x3) {
        auto X01 = X1 - X0;
        auto X02 = X2 - X0;
        auto X03 = X3 - X0;
        zs::vec<T,3,3> DX{
            X01[0],X02[0],X03[0],
            X01[1],X02[1],X03[1],
            X01[2],X02[2],X03[2]};
        
        auto IB = zs::inverse(DX);          
        return deformtion_gradient(x0,x1,x2,x3,IB);
    }



    template<typename T>
    constexpr void deformation_xform(
            const zs::vec<T,3>& X0,const zs::vec<T,3>& X1,const zs::vec<T,3>& X2,const zs::vec<T,3>& X3,
            const zs::vec<T,3>& x0,const zs::vec<T,3>& x1,const zs::vec<T,3>& x2,const zs::vec<T,3>& x3,
            zs::vec<T,3,3>& F,zs::vec<T,3>& b) {
        auto X01 = X1 - X0;
        auto X02 = X2 - X0;
        auto X03 = X3 - X0;
        zs::vec<T,3,3> DX{
            X01[0],X02[0],X03[0],
            X01[1],X02[1],X03[1],
            X01[2],X02[2],X03[2]};
        
        auto IB = zs::inverse(DX);   
        deformation_xform(x0,x1,x2,x3,X0,IB,F,b);
    }

    template<typename T>
    constexpr void deformation_xform(
        const zs::vec<T,3>& x0,const zs::vec<T,3>& x1,const zs::vec<T,3>& x2,const zs::vec<T,3>& x3,
        const zs::vec<T,3>& X0,const zs::vec<T,3,3>& IB,
        zs::vec<T,3,3>& F,zs::vec<T,3>& b) {
            F = deformation_gradient(x0,x1,x2,x3,IB);
            b = x0 - F * X0;
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

};
};