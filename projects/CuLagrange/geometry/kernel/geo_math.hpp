#pragma once

#include <array>
#include <vector>
#include <zensim/math/VecInterface.hpp>
#include <zensim/geometry/Distance.hpp>

namespace zeno { namespace LSL_GEO {

    using REAL = float;
    using VECTOR12 = typename zs::vec<REAL,12>;
    using VECTOR4 = typename zs::vec<REAL,4>;
    using VECTOR3 = typename zs::vec<REAL,3>;
    using VECTOR2 = typename zs::vec<REAL,2>;
    using MATRIX2 = typename zs::vec<REAL,2,2>;
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

    template<typename VecT, zs::enable_if_all<VecT::dim == 1, (VecT::extent <= 3), (VecT::extent > 1)> = 0>
    constexpr auto facet_normal(const zs::VecInterface<VecT>& p0,const zs::VecInterface<VecT>& p1,const zs::VecInterface<VecT>& p2) {
        return (p1 - p0).cross(p2 - p0).normalized();
    }

    template<typename VecT, zs::enable_if_all<VecT::dim == 1, (VecT::extent <= 3), (VecT::extent > 1)> = 0>
    constexpr auto area(const zs::VecInterface<VecT>& p0,const zs::VecInterface<VecT>& p1,const zs::VecInterface<VecT>& p2){
        auto a = (p0 - p1).norm();
        auto b = (p0 - p2).norm();
        auto c = (p1 - p2).norm();
        return doublearea(a,b,c) / (typename VecT::value_type)2.0;
    }

    template<typename VecT, zs::enable_if_all<VecT::dim == 1, (VecT::extent <= 3), (VecT::extent > 1)> = 0>
    constexpr auto cotTheta(const zs::VecInterface<VecT>& e0,const zs::VecInterface<VecT>& e1){
        auto costheta = e0.dot(e1);
        auto sintheta = e0.cross(e1).norm();
        return (costheta / sintheta);
    }

    template<typename DREAL,typename VecT, zs::enable_if_all<VecT::dim == 1, (VecT::extent <= 3), (VecT::extent > 1)> = 0,typename REAL = typename VecT::value_type>
    constexpr auto cotTheta(const zs::VecInterface<VecT>& e0,const zs::VecInterface<VecT>& e1){
        auto de0 = e0.template cast<DREAL>();
        auto de1 = e1.template cast<DREAL>();
        auto costheta = de0.dot(de1);
        auto sintheta = de0.cross(de1).norm();
        return (REAL)(costheta / sintheta);
    }

    template<typename VecT, zs::enable_if_all<VecT::dim == 1, (VecT::extent <= 3), (VecT::extent > 1)> = 0>
    constexpr int orient(const zs::VecInterface<VecT>& p0,
            const zs::VecInterface<VecT>& p1,
            const zs::VecInterface<VecT>& p2,
            const zs::VecInterface<VecT>& p3){ 
        auto nrm = facet_normal(p0,p1,p2);
        auto seg = p3 - p0;
        auto d = nrm.dot(seg);
        return d < 0 ? -1 : 1;        
    }

    template<typename VecT, zs::enable_if_all<VecT::dim == 1, (VecT::extent <= 3), (VecT::extent > 1)> = 0>
    constexpr auto is_inside_tet(const zs::VecInterface<VecT>& p0,
            const zs::VecInterface<VecT>& p1,
            const zs::VecInterface<VecT>& p2,
            const zs::VecInterface<VecT>& p3,
            const zs::VecInterface<VecT>& p){
        auto orient_tet = orient(p0,p1,p2,p3);
        if(orient_tet != orient(p,p1,p2,p3))
            return false;
        if(orient_tet != orient(p0,p,p2,p3))
            return false;
        if(orient_tet != orient(p0,p1,p,p3))
            return false; 
        if(orient_tet != orient(p0,p1,p2,p))
            return false;         

        return true;
    }


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
        const zs::vec<T,3>& x0,const zs::vec<T,3>& x1,const zs::vec<T,3>& x2,const zs::vec<T,3>& x3,
        const zs::vec<T,3>& X0,const zs::vec<T,3,3>& IB,
        zs::vec<T,3,3>& F,zs::vec<T,3>& b) {
            F = deformation_gradient(x0,x1,x2,x3,IB);
            b = x0 - F * X0;
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




    ///////////////////////////////////////////////////////////////////////
    // get the linear interpolation coordinates from v0 to the line segment
    // between v1 and v2
    ///////////////////////////////////////////////////////////////////////
    constexpr VECTOR2 get_lerp(const VECTOR3 v0, const VECTOR3& v1, const VECTOR3& v2)
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
    constexpr REAL point_line_distance(const VECTOR3 v0, const VECTOR3& v1, const VECTOR3& v2)
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
    constexpr VECTOR3 get_vertex_triangle_barycentric_coordinates(const VECTOR3 vertices[4])
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
    constexpr VECTOR3 get_vertex_triangle_inside_barycentric_coordinates(const VECTOR3 vertices[4])
    {
        VECTOR3 barycentric = get_vertex_triangle_barycentric_coordinates(vertices);

        // if it's already inside, we're all done
        if (barycentric[0] >= 0.0 &&
            barycentric[1] >= 0.0 &&
            barycentric[2] >= 0.0)
            return barycentric;

        // find distance to all the line segments
        //
        // there's lots of redundant computation between here and get_lerp,
        // but let's get it working and see if it fixes the actual
        // artifact before optimizing
        REAL distance12 = point_line_distance(vertices[0], vertices[1], vertices[2]);
        REAL distance23 = point_line_distance(vertices[0], vertices[2], vertices[3]);
        REAL distance31 = point_line_distance(vertices[0], vertices[3], vertices[1]);

        // less than or equal is important here, otherwise fallthrough breaks
        if (distance12 <= distance23 && distance12 <= distance31)
        {
            VECTOR2 lerp = get_lerp(vertices[0], vertices[1], vertices[2]);
            barycentric[0] = lerp[0];
            barycentric[1] = lerp[1];
            barycentric[2] = 0.0;
            return barycentric;
        }
        
        // less than or equal is important here, otherwise fallthrough breaks
        if (distance23 <= distance12 && distance23 <= distance31)
        {
            VECTOR2 lerp = get_lerp(vertices[0], vertices[2], vertices[3]);
            barycentric[0] = 0.0;
            barycentric[1] = lerp[0];
            barycentric[2] = lerp[1];
            return barycentric;
        }

        // else it must be the 31 case
        VECTOR2 lerp = get_lerp(vertices[0], vertices[3], vertices[1]);
        barycentric[0] = lerp[1];
        barycentric[1] = 0.0;
        barycentric[2] = lerp[0];
        return barycentric;
    }


///////////////////////////////////////////////////////////////////////
// compute distance between a point and triangle
///////////////////////////////////////////////////////////////////////

constexpr REAL get_vertex_triangle_distance(const VECTOR3& v0, const VECTOR3& v1, 
                                        const VECTOR3& v2, const VECTOR3& v,VECTOR3& barycentric,VECTOR3& project_bary)
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
            project_bary = barycentric;
            // project_v = barycentric[0] * v0 + barycentric[1] * v1 + barycentric[2] * v2;
            return zs::abs(normalDistance);
        }

        VECTOR3 vs[3] = {v0,v1,v2};

        VECTOR3 es[3] = {};

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
        // VECTOR3 projected_e[3] = {};
        if (e1dot > 0.0 && e1dot < e1.norm())
        {
            const VECTOR3 projected = v0 + e1Hat * e1dot;
            es[0] = projected;
            edgeDistances[0] = (v - projected).norm();
        }
        const REAL e2dot = e2Hat.dot(ev);
        if (e2dot > 0.0 && e2dot < e2.norm())
        {
            const VECTOR3 projected = v0 + e2Hat * e2dot;
            es[1] = projected;
            edgeDistances[1] = (v - projected).norm();
        }
        const REAL e3dot = e3Hat.dot(ev3);
        if (e3dot > 0.0 && e3dot < e3.norm())
        {
            const VECTOR3 projected = v1 + e3Hat * e3dot;
            es[2] = projected;
            edgeDistances[2] = (v - projected).norm();
        }

        // get the distance to each vertex
        const VECTOR3 vertexDistances((v - v0).norm(), 
                                        (v - v1).norm(), 
                                        (v - v2).norm());

        // get the smallest of both the edge and vertex distances
        REAL vertexMin = 1e8;
        REAL edgeMin = 1e8;

        int min_e_idx = 0;
        int min_v_idx = 0;
        // vec3 project_v_min{};
        // vec3 project_e_min{};

        for(int i = 0;i < 3;++i){
            if(vertexMin > vertexDistances[i]){
                vertexMin = vertexDistances[i];
                min_v_idx = i;
            }
            if(edgeMin > edgeDistances[i]){
                edgeMin = edgeDistances[i];
                min_e_idx = i;
            }
            // vertexMin = vertexMin > vertexDistances[i] ? vertexDistances[i] : vertexMin;
            // edgeMin = edgeMin > edgeDistances[i] ? edgeDistances[i] : edgeMin;
        }
        VECTOR3 project_v{};
        if(vertexMin < edgeMin)
            project_v = vs[min_v_idx];
        else
            project_v = es[min_e_idx];


        // const VECTOR3 e1 = v1 - v0;
        // const VECTOR3 e2 = v2 - v0;
        // const VECTOR3 n = e1.cross(e2);
        auto na_p = (v2 - v1).cross(project_v - v1);
        auto nb_p = (v0 - v2).cross(project_v - v2);
        auto nc_p = (v1 - v0).cross(project_v - v0);
        project_bary = VECTOR3(n.dot(na_p) / n.l2NormSqr(),
                                    n.dot(nb_p) / n.l2NormSqr(),
                                    n.dot(nc_p) / n.l2NormSqr());

        // return the smallest of those
        return (vertexMin < edgeMin) ? vertexMin : edgeMin;
    }

    constexpr REAL get_vertex_triangle_distance(const VECTOR3& v0, const VECTOR3& v1, 
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

    constexpr REAL get_vertex_triangle_distance(const VECTOR3& v0, const VECTOR3& v1, 
                                        const VECTOR3& v2, const VECTOR3& v)
    {
        VECTOR3 barycentric{};
        return get_vertex_triangle_distance(v0,v1,v2,v,barycentric);
    }


    constexpr void get_triangle_vertex_barycentric_coordinates(const VECTOR3& v1, const VECTOR3& v2, const VECTOR3& v3, const VECTOR3& v4,VECTOR3& bary) {
        constexpr auto eps = 1e-6;
        auto x13 = v1 - v3;
        auto x23 = v2 - v3;
        auto x43 = v4 - v3;
        auto A00 = x13.dot(x13);
        auto A01 = x13.dot(x23);
        auto A11 = x23.dot(x23);
        auto b0 = x13.dot(x43);
        auto b1 = x23.dot(x43);
        auto detA = A00 * A11 - A01 * A01;
        bary[0] = ( A11 * b0 - A01 * b1) / (detA + eps);
        bary[1] = (-A01 * b0 + A00 * b1) / (detA + eps);
        bary[2] = 1 - bary[0] - bary[1];
    }

    constexpr void get_triangle_vertex_barycentric_coordinates(const VECTOR3& v1, const VECTOR3& v2, const VECTOR3& v3, const VECTOR3& v4,VECTOR3& bary,
            REAL& detA,REAL& A00,REAL& A11,REAL& A01) {
        constexpr auto eps = 1e-6;
        auto x13 = v1 - v3;
        auto x23 = v2 - v3;
        auto x43 = v4 - v3;
        A00 = x13.dot(x13);
        A01 = x13.dot(x23);
        A11 = x23.dot(x23);
        auto b0 = x13.dot(x43);
        auto b1 = x23.dot(x43);
        detA = A00 * A11 - A01 * A01;
        bary[0] = ( A11 * b0 - A01 * b1) / (detA + eps);
        bary[1] = (-A01 * b0 + A00 * b1) / (detA + eps);
        bary[2] = 1 - bary[0] - bary[1];
    }

    constexpr void get_edge_edge_barycentric_coordinates(const VECTOR3& v1, const VECTOR3& v2,const VECTOR3& v3, const VECTOR3& v4,VECTOR2& bary) {
        constexpr auto eps = 1e-6;
        auto x21 = v2 - v1;
        auto x43 = v4 - v3;
        auto x31 = v3 - v1;
        auto A00 = x21.dot(x21);
        auto A01 = -x21.dot(x43);
        auto A11 = x43.dot(x43);
        auto b0 = x21.dot(x31);
        auto b1 = -x43.dot(x31);
        auto detA = A00 * A11 - A01 * A01;

        bary[0] = ( A11 * b0 - A01 * b1) / (detA + eps);
        bary[1] = (-A01 * b0 + A00 * b1) / (detA + eps);
        // }else { // the two edge is almost parallel

        // }
    }

    constexpr void get_segment_triangle_intersection_barycentric_coordinates(const VECTOR3& e0,const VECTOR3& e1,
        const VECTOR3& t0,const VECTOR3& t1,const VECTOR3& t2,
        VECTOR2& edge_bary,VECTOR3& tri_bary) {
            auto tnrm = LSL_GEO::facet_normal(t0,t1,t2);
            auto d0 = (e0 - t0).dot(tnrm);
            auto d01 = (e0 - e1).dot(tnrm);

            edge_bary[0] = (1 - d0 / d01);
            edge_bary[1] = d0 / d01;

            auto ep = edge_bary[0] * e0  + edge_bary[1] * e1;


            get_triangle_vertex_barycentric_coordinates(t0,t1,t2,ep,tri_bary);
    }   


    constexpr REAL get_vertex_edge_barycentric_coordinates(const VECTOR3& v1, const VECTOR3& v2,const VECTOR3& v3) {
        auto x32 = v3 - v2;
        // auto nx32 = x32.l2NormSqr();
        auto x12 = v1 - v2;
        // auto dir = (v3 - v2).normalized();
        return x12.dot(x32)/x32.l2NormSqr();
    }

    constexpr REAL get_edge_edge_intersection_barycentric_coordinates(const VECTOR3& v1, const VECTOR3& v2,const VECTOR3& v3, const VECTOR3& v4,VECTOR2& bary,int& type) {
        REAL dist2{zs::limits<REAL>::max()};
        type = ee_distance_type(v1,v2,v3,v4);
        switch (type) {
            case 0:
                dist2 = dist2_pp(v1,v3);
                bary = VECTOR2{0,0};
                break;
            case 1:
                dist2 = dist2_pp(v1,v4);
                bary = VECTOR2{0,1};
                break;
            case 2:
                dist2 = dist2_pe(v1,v3,v4);
                bary = VECTOR2{0,get_vertex_edge_barycentric_coordinates(v1,v3,v4)};
                break;
            case 3:
                dist2 = dist2_pp(v2,v3);
                bary = VECTOR2{1,0};
                break;
            case 4:
                dist2 = dist2_pp(v2,v4);
                bary = VECTOR2{1,1};
                break;
            case 5:
                dist2 = dist2_pe(v2,v3,v4);
                bary = VECTOR2{1,get_vertex_edge_barycentric_coordinates(v2,v3,v4)};
                break;
            case 6:
                dist2 = dist2_pe(v3,v1,v2);
                bary = VECTOR2{get_vertex_edge_barycentric_coordinates(v3,v1,v2),0};
                break;
            case 7:
                dist2 = dist2_pe(v4,v1,v2);
                bary = VECTOR2{get_vertex_edge_barycentric_coordinates(v4,v1,v2),1};
                break;
            case 8:
                dist2 = dist2_ee(v1,v2,v3,v4);
                get_edge_edge_barycentric_coordinates(v1,v2,v3,v4,bary);
                break;
            default:
                break;
        }
        return dist2;
    }


    constexpr REAL get_edge_edge_distance(const VECTOR3& v1, const VECTOR3& v2,const VECTOR3& v3, const VECTOR3& v4) {
        REAL dist2{zs::limits<REAL>::max()};
        auto type = ee_distance_type(v1,v2,v3,v4);
        switch (type) {
            case 0:
                dist2 = dist2_pp(v1,v3);
                // bary = VECTOR2{0,0};
                break;
            case 1:
                dist2 = dist2_pp(v1,v4);
                // bary = VECTOR2{0,1};
                break;
            case 2:
                dist2 = dist2_pe(v1,v3,v4);
                // bary = VECTOR2{0,get_vertex_edge_barycentric_coordinates(v1,v3,v4)};
                break;
            case 3:
                dist2 = dist2_pp(v2,v3);
                // bary = VECTOR2{1,0};
                break;
            case 4:
                dist2 = dist2_pp(v2,v4);
                // bary = VECTOR2{1,1};
                break;
            case 5:
                dist2 = dist2_pe(v2,v3,v4);
                // bary = VECTOR2{1,get_vertex_edge_barycentric_coordinates(v2,v3,v4)};
                break;
            case 6:
                dist2 = dist2_pe(v3,v1,v2);
                // bary = VECTOR2{get_vertex_edge_barycentric_coordinates(v3,v1,v2),0};
                break;
            case 7:
                dist2 = dist2_pe(v4,v1,v2);
                // bary = VECTOR2{get_vertex_edge_barycentric_coordinates(v4,v1,v2),1};
                break;
            case 8:
                dist2 = dist2_ee(v1,v2,v3,v4);
                // get_edge_edge_barycentric_coordinates(v1,v2,v3,v4,bary);
                break;
            default:
                break;
        }
        return zs::sqrt(dist2);
    }

    constexpr REAL get_vertex_triangle_intersection_barycentric_coordinates(const VECTOR3& p, const VECTOR3& t0,const VECTOR3& t1, const VECTOR3& t2,VECTOR3& bary) {
        REAL dist2{zs::limits<REAL>::max()};
        REAL eb{};
        switch (pt_distance_type(p, t0, t1, t2)) {
            case 0:
                dist2 = dist2_pp(p,t0);
                bary = VECTOR3{1,0,0};
                break;
            case 1:
                dist2 = dist2_pp(p,t1);
                bary = VECTOR3{0,1,0};
                break;
            case 2:
                dist2 = dist2_pp(p,t2);
                bary = VECTOR3{0,0,1};
                break;
            case 3:
                dist2 = dist2_pe(p,t0,t1);
                eb = get_vertex_edge_barycentric_coordinates(p,t0,t1);
                bary = VECTOR3{1-eb,eb,0};
                break;
            case 4:
                dist2 = dist2_pe(p,t1,t2);
                eb = get_vertex_edge_barycentric_coordinates(p,t1,t2);
                bary = VECTOR3{0,1-eb,eb};
                break;
            case 5:
                dist2 = dist2_pe(p,t0,t2);
                eb = get_vertex_edge_barycentric_coordinates(p,t0,t2);
                bary = VECTOR3{1-eb,0,eb};
                break;
            case 6:
                dist2 = dist2_pt(p, t0, t1, t2);
                get_triangle_vertex_barycentric_coordinates(t0,t1,t2,p,bary);
                break;
            default:
                break;
        }
        return dist2;
    }

    constexpr REAL get_vertex_triangle_distance(const VECTOR3& v0, const VECTOR3& v1, 
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


    constexpr VECTOR2 removeVecDoF(const VECTOR3& v,int dof) {
        return VECTOR2{v[(dof + 1) % 3],v[(dof + 2) % 3]};
    }



    constexpr bool is_ray_triangle_intersection(const VECTOR3 rayOrigin, 
                            const VECTOR3 rayVector, 
                            const VECTOR3& vertex0,
                            const VECTOR3& vertex1,
                            const VECTOR3& vertex2,
                            const REAL& EPSILON){
        auto edge1 = vertex1 - vertex0;
        auto edge2 = vertex2 - vertex0;
        auto h = rayVector.cross(edge2);
        auto a = edge1.dot(h);

        if (a > -EPSILON && a < EPSILON)
            return false;    // This ray is parallel to this triangle.

        auto f = static_cast<REAL>(1.0) / a;
        auto s = rayOrigin - vertex0;
        auto u = f * s.dot(h);

        if (u < 0.0 || u > 1.0)
            return false;

        auto q = s.cross(edge1);
        auto v = f * rayVector.dot(q);

        if (v < 0.0 || u + v > 1.0)
            return false;

        // At this stage we can compute t to find out where the intersection point is on the line.
        auto t = f * edge2.dot(q);

        if (t > EPSILON) // ray intersection
            return true;
        else // This means that there is a line intersection but not a ray intersection.
            return false;
    }

    ///////////////////////////////////////////////////////////////////////
    // see if the projection of v onto the plane of v0,v1,v2 is inside 
    // the triangle formed by v0,v1,v2
    ///////////////////////////////////////////////////////////////////////
    constexpr bool vertex_projects_inside_triangle(const VECTOR3& v0, const VECTOR3& v1, 
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

    template<typename DREAL>
    constexpr bool tri_ray_intersect_d(VECTOR3 const &from_, VECTOR3 const &to_, VECTOR3 const &v0_, VECTOR3 const &v1_, VECTOR3 const &v2_,DREAL& r_) {
        using DVECTOR3 = zs::vec<DREAL,3>; 

        auto ro = from_.cast<DREAL>();
        auto rd = to_.cast<DREAL>() - ro;
        auto v0 = v0_.cast<DREAL>();
        auto v1 = v1_.cast<DREAL>();
        auto v2 = v2_.cast<DREAL>();

        const DREAL eps = std::numeric_limits<DREAL>::epsilon() * 10;

        DVECTOR3 u = v1 - v0;
        DVECTOR3 v = v2 - v0;
        DVECTOR3 n = cross(u, v);
        // n = n/n.norm();
        DREAL b = dot(n,rd);
        // b = dot(n,rd);
       
        if (zs::abs(b) > eps) {
            DREAL a = dot(n,v0 - ro);
            DREAL r = a / b;
            // rtmp = r;
            if (r > eps) {
                DVECTOR3 ip = ro + r * rd;
                // ip_dist = dot(ip - v0,n);
                DREAL uu = dot(u, u);
                DREAL uv = dot(u, v);
                DREAL vv = dot(v, v);
                DVECTOR3 w = ip - v0;
                DREAL wu = dot(w, u);
                DREAL wv = dot(w, v);
                // DREAL ww = dot(w, w);
                DREAL d = uv * uv - uu * vv;
                DREAL s = uv * wv - vv * wu;
                DREAL t = uv * wu - uu * wv;
                // REAL d = zs::sqrt(uu * vv - uv * uv);
                // REAL s = zs::sqrt(uu * ww - wu * wu);
                // real t = zs::sqrt(ww * vv - wv * wv);

                // area[0] = d;
                // area[1] = s;
                // area[2] = t;

                d = (DREAL)1.0 / d;
                s *= d;
                t *= d;
                // dtmp = d;
                // stmp = s;
                // ttmp = t;

                // bary[0] = (REAL)1.0 - s - t;
                // bary[1] = s;
                // bary[2] = t;
                if (-eps <= s && s <= 1 + eps && -eps <= t && s + t <= 1 + eps * 2){
                    r_ = (DREAL)r;
                    if(r < (DREAL)1.0 + eps && r > (DREAL)0.0 - eps)
                        return true;
                    // else {
                    //     printf("invalid r detected : %f\n",(float)r);
                    //     return false;
                    // }
                    // return true;
                }
            }
        }
        r_ = std::numeric_limits<REAL>::infinity();
        return false;
    }

    constexpr REAL tri_ray_intersect(VECTOR3 const &ro, VECTOR3 const &rd, VECTOR3 const &v0, VECTOR3 const &v1, VECTOR3 const &v2) {
        const REAL eps = std::numeric_limits<REAL>::epsilon() * 10;
        VECTOR3 u = v1 - v0;
        VECTOR3 v = v2 - v0;
        VECTOR3 n = cross(u, v);
        // n = n/n.norm();
        REAL b = dot(n, rd);
        // b = dot(n,rd);
        if (zs::abs(b) > eps) {
            REAL a = n.dot(v0 - ro);
            REAL r = a / b;
            // rtmp = r;
            if (r > eps) {
                VECTOR3 ip = ro + r * rd;
                // ip_dist = dot(ip - v0,n);
                REAL uu = dot(u, u);
                REAL uv = dot(u, v);
                REAL vv = dot(v, v);
                VECTOR3 w = ip - v0;
                REAL wu = dot(w, u);
                REAL wv = dot(w, v);
                REAL ww = dot(w, w);
                REAL d = uv * uv - uu * vv;
                REAL s = uv * wv - vv * wu;
                REAL t = uv * wu - uu * wv;


                d = (REAL)1.0 / d;
                s *= d;
                t *= d;
                // dtmp = d;
                // stmp = s;
                // ttmp = t;

                // bary[0] = (REAL)1.0 - s - t;
                // bary[1] = s;
                // bary[2] = t;
                if (-eps <= s && s <= 1 + eps && -eps <= t && s + t <= 1 + eps * 2)
                    return r;
            }
        }
        return std::numeric_limits<REAL>::infinity();
    }

    constexpr REAL ray_ray_intersect(VECTOR3 const& x0,VECTOR3 const& v0,VECTOR3 const &x1,VECTOR3 const& v1,REAL thickness) {
        auto x =  x1 - x0;
        auto v = v1 - v0;
        
        if(x.norm() < thickness && x.dot(v) > 0)
            return std::numeric_limits<REAL>::infinity();
        if(x.norm() < thickness && x.dot(v) < 0)
            return (REAL)0;
        // if(vv.norm() < 1e)

        auto xx = x.dot(x);
        auto vv = v.dot(v);
        auto xv = x.dot(v);

        // auto closest_dist = (x - xv / vv * v).norm();
        // if(closest_dist > thickness)
        //     return std::numeric_limits<REAL>::infinity();

        auto delta = 4 * xv * xv - 4 * vv * (xx - thickness * thickness);
        if(delta < 0)
            return std::numeric_limits<REAL>::infinity();

        auto sqrt_delta = zs::sqrt(delta);
        auto alpha = xv / vv;
        auto beta = sqrt_delta / 2 / vv;

        auto t0 = alpha + beta;
        auto t1 = alpha - beta;

        t0 = t0 < (REAL)1.0 && t0 > (REAL)0.0 ? t0 : std::numeric_limits<REAL>::infinity();
        t1 = t1 < (REAL)1.0 && t1 > (REAL)0.0 ? t1 : std::numeric_limits<REAL>::infinity(); 

        return zs::min(t0,t1);
    }
};
};