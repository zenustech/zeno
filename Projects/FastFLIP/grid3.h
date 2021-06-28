#ifndef GRID3_H
#define GRID3_H

#include <vec.h>

template<class T>
struct Grid3
{
    Vec<3,T> origin;
    T dx, over_dx;
};

typedef Grid3<float> Grid3f;
typedef Grid3<double> Grid3d;


#endif
