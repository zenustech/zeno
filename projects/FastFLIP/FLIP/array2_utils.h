#ifndef ARRAY2_UTILS_H
#define ARRAY2_UTILS_H

#include "vec.h"
#include "array2.h"
#include "util.h"

template<class S, class T>
T interpolate_value(const Vec<2,S>& point, const Array2<T, Array1<T> >& grid) {
   int i,j;
   S fx,fy;

   get_barycentric(point[0], i, fx, 0, grid.ni);
   get_barycentric(point[1], j, fy, 0, grid.nj);
   
   return bilerp(
      grid(i,j), grid(i+1,j), 
      grid(i,j+1), grid(i+1,j+1), 
      fx, fy);
}

template<class S, class T>
float interpolate_gradient(Vec<2,T>& gradient, const Vec<2,S>& point, const Array2<T, Array1<float> >& grid) {
   int i,j;
   S fx,fy;
   get_barycentric(point[0], i, fx, 0, grid.ni);
   get_barycentric(point[1], j, fy, 0, grid.nj);

   T v00 = grid(i,j);
   T v01 = grid(i,j+1);
   T v10 = grid(i+1,j);
   T v11 = grid(i+1,j+1);

   T ddy0 = (v01 - v00);
   T ddy1 = (v11 - v10);

   T ddx0 = (v10 - v00);
   T ddx1 = (v11 - v01);

   gradient[0] = lerp(ddx0,ddx1,fy);
   gradient[1] = lerp(ddy0,ddy1,fx);
   
   //may as well return value too
   return bilerp(v00, v10, v01, v11, fx, fy);
}

template<class T>
void write_matlab_array(std::ostream &output, Array2<T, Array1<T> >&a, const char *variable_name, bool transpose=false)
{
   output<<variable_name<<"=[";
   for(int j = 0; j < a.nj; ++j) {
      for(int i = 0; i < a.ni; ++i)  {
         output<<a(i,j)<<" ";
      }
      output<<";";
   }
   output<<"]";
   if(transpose)
      output<<"'";
   output<<";"<<std::endl;
}

#endif
