#pragma once

#include <array>
#include <vector>

namespace zeno { namespace LSL_GEO {

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
    template<typename T,typename V = typename zs::vec<T,3>>
    constexpr T area(const V& p0,const V& p1,const V& p2){
        auto a = (p0 - p1).norm();
        auto b = (p0 - p2).norm();
        auto c = (p1 - p2).norm();
        return doublearea(a,b,c) / 2.0;
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

};
};