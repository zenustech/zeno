#ifndef ARRAY3_UTILS_H
#define ARRAY3_UTILS_H

#include <array3.h>
#include <util.h>
#include <vec.h>

namespace LosTopos {

	
template<class S, class T>
T cubic_interpolate_value(const Vec<3,S>& point, const Array3<T, Array1<T> >& grid) 
{
    int i,j,k;
    S fi,fj,fk;
    
    get_barycentric(point[0], i, fi, 0, grid.ni);
    get_barycentric(point[1], j, fj, 0, grid.nj);
    get_barycentric(point[2], k, fk, 0, grid.nk);
    
    return tricubic_interp( grid, i-1, j-1, k-1, fi, fj, fk );
    
}


template<class S, class T>
T interpolate_value(const Vec<3,S>& point, const Array3<T, Array1<T> >& grid) {
    int i,j,k;
    S fi,fj,fk;
    
    get_barycentric(point[0], i, fi, 0, grid.ni);
    get_barycentric(point[1], j, fj, 0, grid.nj);
    get_barycentric(point[2], k, fk, 0, grid.nk);
    
    return trilerp(
                   grid(i,j,k), grid(i+1,j,k), grid(i,j+1,k), grid(i+1,j+1,k), 
                   grid(i,j,k+1), grid(i+1,j,k+1), grid(i,j+1,k+1), grid(i+1,j+1,k+1), 
                   fi,fj,fk);
}

template<class S,class T>
T interpolate_gradient(Vec<3,T>& gradient, const Vec<3,S>& point, const Array3<T, Array1<T> >& grid) {
    int i,j,k;
    S fx,fy,fz;
    
    get_barycentric(point[0], i, fx, 0, grid.ni);
    get_barycentric(point[1], j, fy, 0, grid.nj);
    get_barycentric(point[2], k, fz, 0, grid.nk);
    
    T v000 = grid(i,j,k);
    T v001 = grid(i,j,k+1);
    T v010 = grid(i,j+1,k);
    T v011 = grid(i,j+1,k+1);
    T v100 = grid(i+1,j,k);
    T v101 = grid(i+1,j,k+1);
    T v110 = grid(i+1,j+1,k);
    T v111 = grid(i+1,j+1,k+1);
    
    T ddx00 = (v100 - v000);
    T ddx10 = (v110 - v010);
    T ddx01 = (v101 - v001);
    T ddx11 = (v111 - v011);
    T dv_dx = bilerp(ddx00,ddx10,ddx01,ddx11, fy,fz);
    
    T ddy00 = (v010 - v000);
    T ddy10 = (v110 - v100);
    T ddy01 = (v011 - v001);
    T ddy11 = (v111 - v101);
    T dv_dy = bilerp(ddy00,ddy10,ddy01,ddy11, fx,fz);
    
    T ddz00 = (v001 - v000);
    T ddz10 = (v101 - v100);
    T ddz01 = (v011 - v010);
    T ddz11 = (v111 - v110);
    T dv_dz = bilerp(ddz00,ddz10,ddz01,ddz11, fx,fy);
    
    gradient[0] = dv_dx;
    gradient[1] = dv_dy;
    gradient[2] = dv_dz;
    
    //return value for good measure.
    return trilerp(
                   v000, v100,
                   v010, v110, 
                   v001, v101,
                   v011, v111,
                   fx, fy, fz);
}

}
#endif
